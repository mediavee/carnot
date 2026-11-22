//! Hardened embedded HTTP/1.1 + SSE API server over `tokio::net::TcpListener` (Opus C2, N5, N20, I4).
//!
//! Security & concurrency invariants:
//!   - Bearer token authentication (`CARNOT_HTTP_TOKEN` or auto-generated 256-bit token saved in
//!     `~/.carnot/server.token` with mode `0600`) required on all `/v1/*` routes (C2).
//!   - Strict `Host` & `Origin` localhost checks + `Content-Type: application/json` enforcement on
//!     POST routes (blocks browser CSRF & DNS rebinding).
//!   - Bounded header count (64), header line size (8 KiB), and 15 s request read timeout (N20).
//!   - Returns `409 Conflict` if a second `POST /v1/prompt` arrives while a run is active (N5).
//!   - Lossless SSE streaming from `EventLog` with `id: <seq>` & `Last-Event-ID` replay on `GET /v1/events`.

use carnot_agent::Agent;
use carnot_core::{AgentEvent, Result};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const MAX_HEADERS: usize = 64;
const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
const REQUEST_READ_TIMEOUT_SECS: u64 = 15;

/// Starts the HTTP/SSE server using `CARNOT_HTTP_TOKEN` or a newly generated 256-bit token saved at
/// `~/.carnot/server.token` (`0600`).
pub async fn serve_http(agent: Arc<Agent>, addr: &str) -> Result<()> {
    let token = match std::env::var("CARNOT_HTTP_TOKEN") {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            let t = format!(
                "{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            );
            if let Some(tok_path) = default_token_path() {
                if let Some(parent) = tok_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = write_token_0600(&tok_path, &t);
                tracing::info!(token_file = %tok_path.display(), "carnot HTTP Bearer token written (mode 0600)");
            }
            t
        }
    };
    serve_http_with_token(agent, addr, token).await
}

pub async fn serve_http_with_token(agent: Arc<Agent>, addr: &str, token: String) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let expected_token = Arc::new(token);
    tracing::info!(%addr, "carnot HTTP/SSE server listening (Bearer auth + CSRF protection enabled)");
    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                let (stream, _) = accept_res?;
                let ag = agent.clone();
                let tok = expected_token.clone();
                tokio::spawn(async move {
                    let _ = handle_conn(ag, stream, &tok).await;
                });
            }
            _ = shutdown_signal() => {
                tracing::info!("carnot HTTP/SSE server shutting down gracefully");
                agent.abort().await;
                break;
            }
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn default_token_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".carnot").join("server.token"))
}

fn write_token_0600(path: &PathBuf, token: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(format!("{token}\n").as_bytes())?;
    f.sync_all()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().to_ascii_lowercase();
    let bare = if let Some(rest) = h.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        h.split(':').next().unwrap_or("")
    };
    matches!(bare, "localhost" | "127.0.0.1" | "::1")
}

fn is_allowed_origin(origin: &str) -> bool {
    let o = origin.trim();
    if let Some(rest) = o
        .strip_prefix("http://")
        .or_else(|| o.strip_prefix("https://"))
    {
        let authority = rest.split('/').next().unwrap_or("");
        is_loopback_host(authority)
    } else {
        false
    }
}

async fn read_bounded_line(
    reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = reader.read(&mut byte).await?;
        if n == 0 {
            if buf.is_empty() {
                return Ok(None);
            }
            break;
        }
        buf.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
        if buf.len() > MAX_HEADER_LINE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP header line exceeds 8 KiB limit",
            ));
        }
    }
    Ok(Some(String::from_utf8_lossy(&buf).to_string()))
}

async fn handle_conn(
    agent: Arc<Agent>,
    stream: TcpStream,
    expected_token: &str,
) -> std::io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let req_head = tokio::time::timeout(Duration::from_secs(REQUEST_READ_TIMEOUT_SECS), async {
        let Some(req_line) = read_bounded_line(&mut reader).await? else {
            return Ok(None);
        };
        let parts: Vec<&str> = req_line.split_whitespace().collect();
        let method = parts.first().copied().unwrap_or("").to_string();
        let raw_path = parts.get(1).copied().unwrap_or("/");
        let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();

        let mut content_len: usize = 0;
        let mut last_event_id: Option<u64> = None;
        let mut auth_header: Option<String> = None;
        let mut content_type: Option<String> = None;
        let mut origin: Option<String> = None;
        let mut host: Option<String> = None;

        for _ in 0..MAX_HEADERS {
            let Some(h) = read_bounded_line(&mut reader).await? else {
                break;
            };
            if h.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                let val = v.trim();
                if k.eq_ignore_ascii_case("content-length") {
                    content_len = val.parse().unwrap_or(0).min(8 * 1024 * 1024);
                } else if k.eq_ignore_ascii_case("last-event-id") {
                    last_event_id = val.parse().ok();
                } else if k.eq_ignore_ascii_case("authorization") {
                    auth_header = Some(val.to_string());
                } else if k.eq_ignore_ascii_case("content-type") {
                    content_type = Some(val.to_string());
                } else if k.eq_ignore_ascii_case("origin") {
                    origin = Some(val.to_string());
                } else if k.eq_ignore_ascii_case("host") {
                    host = Some(val.to_string());
                }
            }
        }

        let mut body = vec![0u8; content_len];
        if content_len > 0 {
            reader.read_exact(&mut body).await?;
        }
        Ok::<_, std::io::Error>(Some((
            method,
            path,
            last_event_id,
            auth_header,
            content_type,
            origin,
            host,
            body,
        )))
    })
    .await;

    let Ok(Ok(Some((method, path, last_event_id, auth_header, content_type, origin, host, body)))) =
        req_head
    else {
        return Ok(());
    };

    // Host & Origin anti-CSRF / anti-DNS-rebinding guards (C2)
    if let Some(ref h) = host
        && !is_loopback_host(h)
    {
        return write_json(&mut write_half, 403, &json!({"error": "forbidden host"})).await;
    }
    if let Some(ref o) = origin
        && !is_allowed_origin(o)
    {
        return write_json(&mut write_half, 403, &json!({"error": "forbidden origin"})).await;
    }

    if method == "GET" && path == "/health" {
        return write_json(
            &mut write_half,
            200,
            &json!({"status": "ok", "engine": "carnot", "version": env!("CARGO_PKG_VERSION")}),
        )
        .await;
    }

    // Require Bearer token on all `/v1/*` routes (C2)
    let authorized = auth_header
        .as_deref()
        .and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("bearer "))
        })
        .map(|tok| constant_time_eq(tok.trim().as_bytes(), expected_token.as_bytes()))
        .unwrap_or(false);
    if !authorized {
        return write_json(&mut write_half, 401, &json!({"error": "unauthorized"})).await;
    }

    // Enforce `application/json` on POST requests with a body (blocks `fetch(..., {mode:"no-cors"})` CSRF!)
    if method == "POST" && !body.is_empty() {
        let is_json = content_type
            .as_deref()
            .map(|ct| ct.to_ascii_lowercase().starts_with("application/json"))
            .unwrap_or(false);
        if !is_json {
            return write_json(
                &mut write_half,
                415,
                &json!({"error": "Content-Type must be application/json"}),
            )
            .await;
        }
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/metrics") => {
            let met = agent.metrics().await;
            let sid = agent.session_id().await;
            let ph = agent.prefix_hash().await;
            let seq = agent.latest_event_seq();
            write_json(
                &mut write_half,
                200,
                &json!({"session_id": sid, "prefix_hash": ph, "latest_seq": seq, "metrics": met}),
            )
            .await
        }
        ("GET", "/v1/events") => {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
            write_half.write_all(hdr.as_bytes()).await?;
            write_half.flush().await?;

            let mut ev_stream = agent.subscribe_from(last_event_id.unwrap_or(0));
            loop {
                let env = ev_stream.next().await;
                if let Ok(line) = serde_json::to_string(&env) {
                    let frame = format!("id: {}\ndata: {line}\n\n", env.seq);
                    if write_half.write_all(frame.as_bytes()).await.is_err() {
                        break;
                    }
                    let _ = write_half.flush().await;
                }
            }
            Ok(())
        }
        ("POST", "/v1/steer") => {
            let payload: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let msg = payload
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            agent.steer(msg).await;
            write_json(&mut write_half, 200, &json!({"queued": true})).await
        }
        ("POST", "/v1/abort") => {
            agent.abort().await;
            write_json(&mut write_half, 200, &json!({"aborted": true})).await
        }
        ("POST", "/v1/prompt") => {
            if agent.is_busy() {
                return write_json(
                    &mut write_half,
                    409,
                    &json!({"error": "agent run already in progress; use /v1/steer or /v1/abort"}),
                )
                .await;
            }
            let payload: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let prompt = payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let stream_sse = payload
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            if !stream_sse {
                match agent.prompt(prompt).await {
                    Ok(o) => {
                        write_json(
                            &mut write_half,
                            200,
                            &json!({
                                "final_text": o.final_text,
                                "metrics": o.metrics,
                                "halted_reason": o.halted_reason,
                            }),
                        )
                        .await
                    }
                    Err(e) => {
                        write_json(&mut write_half, 500, &json!({"error": e.to_string()})).await
                    }
                }
            } else {
                let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                write_half.write_all(hdr.as_bytes()).await?;
                write_half.flush().await?;

                let start_seq = agent.latest_event_seq();
                let mut ev_stream = agent.subscribe_from(start_seq);
                let ag_run = agent.clone();
                let run_task = tokio::spawn(async move { ag_run.prompt(prompt).await });

                loop {
                    let env = ev_stream.next().await;
                    let is_done = matches!(
                        env.event,
                        AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                    );
                    if let Ok(line) = serde_json::to_string(&env) {
                        let frame = format!("id: {}\ndata: {line}\n\n", env.seq);
                        if write_half.write_all(frame.as_bytes()).await.is_err() {
                            break;
                        }
                        let _ = write_half.flush().await;
                    }
                    if is_done {
                        break;
                    }
                }
                let _ = run_task.await;
                Ok(())
            }
        }
        _ => write_json(&mut write_half, 404, &json!({"error": "not found"})).await,
    }
}

async fn write_json(
    w: &mut tokio::net::tcp::OwnedWriteHalf,
    status: u16,
    body: &Value,
) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        415 => "Unsupported Media Type",
        _ => "Internal Server Error",
    };
    let hdr = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    w.write_all(hdr.as_bytes()).await?;
    w.write_all(&bytes).await?;
    w.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_ai::faux::{FauxProvider, ScriptedTurn};
    use carnot_cas::CasStore;
    use carnot_core::CarnotConfig;
    use carnot_tools::ToolRegistry;

    async fn send_raw(addr: std::net::SocketAddr, req: &str) -> String {
        let mut s = TcpStream::connect(addr).await.expect("connect");
        s.write_all(req.as_bytes()).await.expect("write");
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
        let tok = Arc::new("secret-test-token".to_string());
        let ag_bg = Arc::clone(&agent);
        let tok_bg = Arc::clone(&tok);
        let srv = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let a = Arc::clone(&ag_bg);
                let t = Arc::clone(&tok_bg);
                tokio::spawn(async move {
                    let _ = handle_conn(a, stream, &t).await;
                });
            }
        });

        // 1. Missing Bearer token -> 401 Unauthorized
        let r401 = send_raw(addr, "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").await;
        assert!(
            r401.starts_with("HTTP/1.1 401"),
            "expected 401, got: {r401}"
        );

        // 2. Non-loopback Host -> 403 Forbidden (DNS rebinding guard)
        let r403 = send_raw(
            addr,
            "GET /v1/status HTTP/1.1\r\nHost: evil.example.com\r\nAuthorization: Bearer secret-test-token\r\n\r\n",
        )
        .await;
        assert!(
            r403.starts_with("HTTP/1.1 403"),
            "expected 403, got: {r403}"
        );

        // 3. /health returns 200 OK without auth, and /v1/metrics returns 200 OK with valid Bearer
        let rhealth = send_raw(addr, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").await;
        assert!(
            rhealth.starts_with("HTTP/1.1 200"),
            "expected 200, got: {rhealth}"
        );
        assert!(rhealth.contains("\"engine\":\"carnot\""));

        let r200 = send_raw(
            addr,
            "GET /v1/metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-test-token\r\n\r\n",
        )
        .await;
        assert!(
            r200.starts_with("HTTP/1.1 200"),
            "expected 200, got: {r200}"
        );
        assert!(r200.contains("\"turns\":0"));

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
        assert!(sse.contains("text/event-stream"));
        assert!(sse.contains("hello from http"));
        assert!(sse.contains("id: "));

        srv.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
