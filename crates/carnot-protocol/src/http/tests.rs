use super::*;
use carnot_ai::faux::{FauxProvider, ScriptedTurn};
use carnot_cas::CasStore;
use carnot_core::CarnotConfig;
use carnot_tools::ToolRegistry;
use std::fmt::Write as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn send_raw(addr: std::net::SocketAddr, req: &str) -> String {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(req.as_bytes()).await.expect("write");
    let _ = s.shutdown().await;
    let mut out = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(2), s.read_to_end(&mut out)).await;
    String::from_utf8_lossy(&out).to_string()
}

#[tokio::test]
async fn http_server_enforces_bearer_auth_csrf_and_streams_events() {
    let dir = std::env::temp_dir().join(format!("carnot-http-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "hello from http",
    )]));
    let agent = Arc::new(Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent"));

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let tok: Arc<str> = Arc::from("secret-test-token");
    let broker = ChannelInteractionBroker::new(Arc::clone(&agent));
    let ag_bg = Arc::clone(&agent);
    let br_bg = Arc::clone(&broker);
    let tok_bg = Arc::clone(&tok);
    let pgate_bg = Arc::new(tokio::sync::Mutex::new(()));
    let srv = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let a = Arc::clone(&ag_bg);
            let b = Arc::clone(&br_bg);
            let t = Arc::clone(&tok_bg);
            let pg = Arc::clone(&pgate_bg);
            tokio::spawn(async move {
                serve_conn(a, b, pg, stream, t).await;
            });
        }
    });

    // 1. Missing Bearer token -> 401 Unauthorized
    let r401 = send_raw(addr, "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").await;
    assert!(
        r401.starts_with("HTTP/1.1 401"),
        "expected 401, got: {r401}"
    );

    // 2. Non-loopback Host & spoofed loopback authorities -> 403 Forbidden (DNS rebinding / CSRF guard)
    for bad_host in [
        "evil.example.com",
        "127.0.0.1:80@evil.com",
        "[::1]evil.com",
        "localhost.evil.com",
    ] {
        let r403 = send_raw(
            addr,
            &format!(
                "GET /v1/status HTTP/1.1\r\nHost: {bad_host}\r\nAuthorization: Bearer secret-test-token\r\n\r\n"
            ),
        )
        .await;
        assert!(
            r403.starts_with("HTTP/1.1 403"),
            "expected 403 for Host `{bad_host}`, got: {r403}"
        );
    }
    assert!(
        !is_allowed_origin("http://127.0.0.1:80@evil.com"),
        "spoofed userinfo origin must be denied"
    );
    assert!(
        !is_allowed_origin("http://[::1]evil.com"),
        "spoofed ipv6 origin must be denied"
    );
    assert!(
        is_allowed_origin("http://127.0.0.1:7744"),
        "loopback ipv4 origin must be allowed"
    );
    assert!(
        is_allowed_origin("http://[::1]:7744"),
        "loopback ipv6 origin must be allowed"
    );
    assert!(
        is_allowed_origin("https://localhost"),
        "localhost origin must be allowed"
    );

    // 2b. >MAX_HEADERS -> 431 Request Header Fields Too Large
    let mut many_hdrs = String::from("GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n");
    for i in 0..70 {
        let _ = write!(many_hdrs, "X-Pad-{i}: v\r\n");
    }
    many_hdrs.push_str("\r\n");
    let r431 = send_raw(addr, &many_hdrs).await;
    assert!(
        r431.starts_with("HTTP/1.1 431"),
        "expected 431 for >64 headers, got: {r431}"
    );

    // 3. /health returns 200 OK without auth, and /v1/metrics returns 200 OK with valid Bearer
    let rhealth = send_raw(addr, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").await;
    assert!(
        rhealth.starts_with("HTTP/1.1 200"),
        "expected 200, got: {rhealth}"
    );
    assert!(
        rhealth.contains("\"engine\":\"carnot\""),
        "health response must identify carnot"
    );

    let r200 = send_raw(
        addr,
        "GET /v1/metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\n\r\n",
    )
    .await;
    assert!(
        r200.starts_with("HTTP/1.1 200"),
        "expected 200, got: {r200}"
    );
    assert!(
        r200.contains("\"turns\":0"),
        "initial metrics must show 0 turns"
    );

    // 4. POST /v1/prompt with stream=true streams SSE frames with monotone `id:` and `hello from http`
    let body = r#"{"prompt":"hi","stream":true}"#;
    let prompt_req = format!(
        "POST /v1/prompt HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let sse = send_raw(addr, &prompt_req).await;
    assert!(
        sse.starts_with("HTTP/1.1 200"),
        "expected 200 SSE, got: {sse}"
    );
    assert!(
        sse.contains("text/event-stream"),
        "SSE response must have text/event-stream content-type"
    );
    assert!(
        sse.contains("hello from http"),
        "SSE stream must contain model response"
    );
    assert!(
        sse.contains("id: "),
        "SSE stream must emit monotone event IDs"
    );

    // 5. Invalid Content-Length -> 400 Bad Request, oversized Content-Length -> 413 Payload Too Large
    let r400 = send_raw(
        addr,
        "POST /v1/steer HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\nContent-Type: application/json\r\nContent-Length: notanumber\r\n\r\n",
    )
    .await;
    assert!(
        r400.starts_with("HTTP/1.1 400"),
        "expected 400, got: {r400}"
    );

    let r413 = send_raw(
        addr,
        "POST /v1/steer HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\nContent-Type: application/json\r\nContent-Length: 99999999\r\n\r\n",
    )
    .await;
    assert!(
        r413.starts_with("HTTP/1.1 413"),
        "expected 413, got: {r413}"
    );

    // 6. Transfer-Encoding: chunked is natively supported by hyper HTTP/1.1
    let chunk_body = r#"{"message":"chunked-steer"}"#;
    let r_chunked = send_raw(
        addr,
        &format!(
            "POST /v1/steer HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{chunk_body}\r\n0\r\n\r\n",
            chunk_body.len()
        ),
    )
    .await;
    assert!(
        r_chunked.starts_with("HTTP/1.1 200"),
        "expected 200 for chunked POST /v1/steer, got: {r_chunked}"
    );
    assert!(
        r_chunked.contains("\"queued\":true"),
        "chunked POST /v1/steer must queue steer message, got: {r_chunked}"
    );

    // 7. POST /v1/undo returns 200 with reverted: null when stack is empty
    let rundo = send_raw(
        addr,
        "POST /v1/undo HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\n\r\n",
    )
    .await;
    assert!(
        rundo.starts_with("HTTP/1.1 200"),
        "expected 200, got: {rundo}"
    );
    assert!(
        rundo.contains("\"reverted\":null"),
        "empty undo stack must return reverted: null"
    );

    srv.abort();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constant_time_eq_handles_equal_mismatch_length_and_empty() {
    assert!(constant_time_eq(b"", b""), "empty slices must match");
    assert!(
        constant_time_eq(b"secret-token-123", b"secret-token-123"),
        "identical tokens must match"
    );
    assert!(
        !constant_time_eq(b"secret-token-123", b"secret-token-124"),
        "last-byte mismatch must fail"
    );
    assert!(
        !constant_time_eq(b"secret-token-123", b"secret-token-12"),
        "length mismatch must fail"
    );
}

#[tokio::test]
async fn http_server_handles_fragmented_tcp_writes_csrf_origins_oversized_headers_and_sse_replay() {
    let dir = std::env::temp_dir().join(format!("carnot-http-hardened-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("fragmented ok")]));
    let agent = Arc::new(Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent"));

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let tok: Arc<str> = Arc::from("hardened-test-token");
    let broker = ChannelInteractionBroker::new(Arc::clone(&agent));
    let ag_bg = Arc::clone(&agent);
    let br_bg = Arc::clone(&broker);
    let tok_bg = Arc::clone(&tok);
    let pgate_bg = Arc::new(tokio::sync::Mutex::new(()));
    let srv = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let a = Arc::clone(&ag_bg);
            let b = Arc::clone(&br_bg);
            let t = Arc::clone(&tok_bg);
            let pg = Arc::clone(&pgate_bg);
            tokio::spawn(async move {
                serve_conn(a, b, pg, stream, t).await;
            });
        }
    });

    // 1. Fragmented TCP write (3 bytes at a time across flushes) still parses cleanly.
    let body = r#"{"prompt":"frag"}"#;
    let raw_req = format!(
        "POST /v1/prompt HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer hardened-test-token\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut s = TcpStream::connect(addr).await.expect("connect");
    for chunk in raw_req.as_bytes().chunks(3) {
        s.write_all(chunk).await.expect("write chunk");
        s.flush().await.expect("flush chunk");
    }
    let _ = s.shutdown().await;
    let mut frag_resp = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(3), s.read_to_end(&mut frag_resp)).await;
    let frag_str = String::from_utf8_lossy(&frag_resp);
    assert!(
        frag_str.starts_with("HTTP/1.1 200"),
        "fragmented POST /v1/prompt must return 200, got: {frag_str}"
    );
    assert!(
        frag_str.contains("fragmented ok"),
        "fragmented response must contain model text, got: {frag_str}"
    );

    // 2. Bad Origin header on valid loopback Host -> 403 Forbidden (CSRF).
    let r_origin = send_raw(
        addr,
        "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://attacker.example\r\nAuthorization: Bearer hardened-test-token\r\n\r\n",
    )
    .await;
    assert!(
        r_origin.starts_with("HTTP/1.1 403"),
        "evil Origin must be rejected with 403, got: {r_origin}"
    );

    // 3. Missing Host header in HTTP/1.1 -> 400 Bad Request (fail-closed).
    let r_no_host = send_raw(
        addr,
        "GET /v1/status HTTP/1.1\r\nAuthorization: Bearer hardened-test-token\r\n\r\n",
    )
    .await;
    assert!(
        r_no_host.starts_with("HTTP/1.1 400"),
        "missing Host header must return 400, got: {r_no_host}"
    );

    // 4. Wrong Content-Type on POST with JSON payload -> 415 Unsupported Media Type.
    let r_ct = send_raw(
        addr,
        "POST /v1/steer HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer hardened-test-token\r\nContent-Type: text/plain\r\nContent-Length: 14\r\n\r\n{\"text\":\"abc\"}",
    )
    .await;
    assert!(
        r_ct.starts_with("HTTP/1.1 415"),
        "non-JSON Content-Type on POST must return 415, got: {r_ct}"
    );

    // 5. Oversized header buffer (>16 KiB) -> 431 Request Header Fields Too Large.
    let huge_hdr = format!("X-Big: {}", "a".repeat(MAX_HEADER_BUF_BYTES + 128));
    let r_huge_hdr = send_raw(
        addr,
        &format!(
            "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n{huge_hdr}\r\nAuthorization: Bearer hardened-test-token\r\n\r\n"
        ),
    )
    .await;
    assert!(
        r_huge_hdr.starts_with("HTTP/1.1 431"),
        "oversized header buffer must return 431, got: {r_huge_hdr}"
    );

    // 6. Malformed request line -> 400 Bad Request (never panics).
    let r_malformed = send_raw(addr, "NOT_AN_HTTP_LINE\r\n\r\n").await;
    assert!(
        r_malformed.starts_with("HTTP/1.1 400"),
        "malformed request line must fail closed with 400, got: {r_malformed}"
    );

    // 7. Unknown route with valid auth -> 404 Not Found.
    let r_404 = send_raw(
        addr,
        "GET /v1/does-not-exist HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer hardened-test-token\r\n\r\n",
    )
    .await;
    assert!(
        r_404.starts_with("HTTP/1.1 404"),
        "unknown route must return 404, got: {r_404}"
    );

    // 8. GET /v1/events with Last-Event-ID replays only events strictly greater than the cursor.
    let last_seq = agent.latest_event_seq();
    assert!(
        last_seq >= 2,
        "prompt run must have emitted at least 2 events"
    );
    let replay_cursor = last_seq.saturating_sub(1);
    let mut sse_s = TcpStream::connect(addr).await.expect("connect sse");
    let sse_req = format!(
        "GET /v1/events HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer hardened-test-token\r\nLast-Event-ID: {replay_cursor}\r\n\r\n"
    );
    sse_s
        .write_all(sse_req.as_bytes())
        .await
        .expect("write sse");
    let expected_id = format!("id: {last_seq}\n");
    let mut sse_buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            break;
        }
        match tokio::time::timeout(deadline - now, sse_s.read(&mut chunk)).await {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(n)) => {
                sse_buf.extend_from_slice(&chunk[..n]);
                if String::from_utf8_lossy(&sse_buf).contains(&expected_id) {
                    break;
                }
            }
            Ok(Err(e)) => panic!("sse read error: {e}"),
        }
    }
    let sse_head = String::from_utf8_lossy(&sse_buf);
    assert!(
        sse_head.starts_with("HTTP/1.1 200"),
        "GET /v1/events must return 200, got: {sse_head}"
    );
    assert!(
        sse_head.contains(&expected_id),
        "SSE replay must include event {last_seq}, got: {sse_head}"
    );
    assert!(
        !sse_head.contains(&format!("id: {replay_cursor}\n")),
        "SSE replay must NOT re-emit event {replay_cursor}, got: {sse_head}"
    );

    srv.abort();
    let _ = std::fs::remove_dir_all(&dir);
}
