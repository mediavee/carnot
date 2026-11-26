use super::pkce::wait_for_loopback_callback;
use super::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;
use uuid::Uuid;

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[test]
fn sha256_matches_nist_and_rfc7636_pkce_s256_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // RFC 7636 Appendix B official test vector:
    assert_eq!(
        pkce_s256_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    );
}

#[test]
fn pkce_pair_and_state_meet_rfc7636_lengths() {
    let (v, c) = generate_pkce_pair();
    assert_eq!(v.len(), 64);
    assert_eq!(c, pkce_s256_challenge(&v));
    let st = generate_oauth_state();
    assert!(st.len() >= 40);
    assert_ne!(v, st);
}

#[test]
fn auth_store_atomic_roundtrip_and_redacted_debug() {
    let dir = std::env::temp_dir().join(format!("carnot-oauth-test-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));

    let saved =
        save_manual_token("openrouter", "sk-or-secret-token-123", &store).expect("save token");
    let dbg = format!("{saved:?}");
    assert!(!dbg.contains("sk-or-secret-token-123"));
    assert!(dbg.contains("<redacted>"));

    let loaded = store.get("openrouter").expect("read").expect("present");
    assert_eq!(loaded.access_token, "sk-or-secret-token-123");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(store.path())
            .expect("meta")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    assert!(store.remove("openrouter").expect("remove"));
    assert!(store.get("openrouter").expect("read").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn standard_pkce_login_and_refresh_end_to_end() {
    // Start a mock OAuth token endpoint on 127.0.0.1.
    let mock_listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind mock");
    let mock_addr = mock_listener.local_addr().expect("addr");
    let server_task = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut s, _) = mock_listener.accept().await.expect("accept");
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).await.expect("read");
            let req = String::from_utf8_lossy(&buf[..n]);
            let body = if attempt == 0 {
                assert!(req.contains("grant_type=authorization_code"));
                assert!(req.contains("code=test-auth-code"));
                assert!(req.contains("code_verifier="));
                r#"{"access_token":"tok-initial","refresh_token":"ref-1","expires_in":1,"token_type":"Bearer"}"#
            } else {
                assert!(req.contains("grant_type=refresh_token"));
                assert!(req.contains("refresh_token=ref-1"));
                r#"{"access_token":"tok-refreshed","expires_in":3600,"token_type":"Bearer"}"#
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes()).await.expect("write");
        }
    });

    let dir = std::env::temp_dir().join(format!("carnot-oauth-pkce-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));
    let cfg = OAuthConfig {
        flow: OAuthFlow::Pkce,
        authorize_url: Some("https://auth.example.test/authorize".into()),
        token_url: Some(format!("http://{mock_addr}/token")),
        client_id: Some("test-client".into()),
        scopes: vec!["read".into(), "write".into()],
        ..Default::default()
    };
    let opts = OAuthLoginOptions {
        open_browser: false,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    let tok = login_with_oauth(
        "custom-oidc",
        &cfg,
        &store,
        &opts,
        |prompt| {
            if let OAuthPrompt::OpenUrl {
                url, callback_url, ..
            } = prompt
            {
                let parsed = Url::parse(&url).expect("valid auth url");
                let params: BTreeMap<String, String> = parsed.query_pairs().into_owned().collect();
                let st = params.get("state").cloned().expect("state");
                let hit_url = format!("{callback_url}?code=test-auth-code&state={st}");
                tokio::spawn(async move {
                    let _ = reqwest::get(&hit_url).await;
                });
            }
        },
        || unreachable!(),
    )
    .await
    .expect("pkce login");

    assert_eq!(tok.access_token, "tok-initial");
    assert_eq!(tok.refresh_token.as_deref(), Some("ref-1"));

    // Now verify automatic proactive refresh (expires_in=1 < skew 120s) preserves refresh_token!
    let mgr = OAuthTokenManager::new("custom-oidc", tok, Some(cfg), store.clone());
    let fresh = mgr.access_token(None).await.expect("refresh");
    assert_eq!(fresh, "tok-refreshed");

    let persisted = store.get("custom-oidc").expect("read").expect("present");
    assert_eq!(persisted.access_token, "tok-refreshed");
    assert_eq!(persisted.refresh_token.as_deref(), Some("ref-1"));

    server_task.await.expect("mock done");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn pkce_key_exchange_end_to_end() {
    let mock_listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let mock_addr = mock_listener.local_addr().expect("addr");
    let server_task = tokio::spawn(async move {
        let (mut s, _) = mock_listener.accept().await.expect("accept");
        let mut buf = [0u8; 4096];
        let n = s.read(&mut buf).await.expect("read");
        let req = String::from_utf8_lossy(&buf[..n]);
        assert!(req.contains("\"code\":\"or-code-99\""));
        assert!(req.contains("\"code_challenge_method\":\"S256\""));
        let body = r#"{"key":"sk-test-v1-live-key"}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        s.write_all(resp.as_bytes()).await.expect("write");
    });

    let dir = std::env::temp_dir().join(format!("carnot-oauth-kx-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));
    let cfg = OAuthConfig {
        flow: OAuthFlow::PkceKeyExchange,
        authorize_url: Some("https://auth.example.test/auth".into()),
        exchange_url: Some(format!("http://{mock_addr}/api/v1/auth/keys")),
        ..Default::default()
    };
    let opts = OAuthLoginOptions {
        open_browser: false,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    let tok = login_with_oauth(
        "gateway-kx",
        &cfg,
        &store,
        &opts,
        |prompt| {
            if let OAuthPrompt::OpenUrl { callback_url, .. } = prompt {
                let hit_url = format!("{callback_url}?code=or-code-99");
                tokio::spawn(async move {
                    let _ = reqwest::get(&hit_url).await;
                });
            }
        },
        || unreachable!(),
    )
    .await
    .expect("pkce key exchange");

    assert_eq!(tok.access_token, "sk-test-v1-live-key");
    server_task.await.expect("server");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn device_code_flow_handles_pending_and_succeeds() {
    let mock_listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let mock_addr = mock_listener.local_addr().expect("addr");
    let server_task = tokio::spawn(async move {
        for step in 0..3 {
            let (mut s, _) = mock_listener.accept().await.expect("accept");
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf).await.expect("read");
            let (status, body) = match step {
                0 => (
                    "200 OK",
                    r#"{"device_code":"dev-123","user_code":"ABCD-EFGH","verification_uri":"https://auth.example.test/device","expires_in":60,"interval":1}"#,
                ),
                1 => ("400 Bad Request", r#"{"error":"authorization_pending"}"#),
                _ => (
                    "200 OK",
                    r#"{"access_token":"tok-device-ok","refresh_token":"ref-dev","expires_in":7200}"#,
                ),
            };
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes()).await.expect("write");
        }
    });

    let dir = std::env::temp_dir().join(format!("carnot-oauth-dev-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));
    let cfg = OAuthConfig {
        flow: OAuthFlow::DeviceCode,
        device_authorization_url: Some(format!("http://{mock_addr}/device")),
        token_url: Some(format!("http://{mock_addr}/token")),
        client_id: Some("dev-client".into()),
        ..Default::default()
    };
    let opts = OAuthLoginOptions {
        open_browser: false,
        timeout: Duration::from_secs(15),
        ..Default::default()
    };
    let mut saw_user_code = false;
    let tok = login_with_oauth(
        "device-prov",
        &cfg,
        &store,
        &opts,
        |prompt| {
            if let OAuthPrompt::DeviceCode { user_code, .. } = prompt {
                assert_eq!(user_code, "ABCD-EFGH");
                saw_user_code = true;
            }
        },
        || unreachable!(),
    )
    .await
    .expect("device flow");

    assert!(saw_user_code);
    assert_eq!(tok.access_token, "tok-device-ok");
    assert_eq!(tok.refresh_token.as_deref(), Some("ref-dev"));
    server_task.await.expect("server");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn loopback_callback_respects_timeout_even_with_idle_socket() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    // Open an idle socket that sends nothing (simulates Chrome speculative preconnect / stall).
    let _idle_socket = tokio::net::TcpStream::connect(addr).await.expect("connect");

    let start = tokio::time::Instant::now();
    let res = wait_for_loopback_callback(
        listener,
        "/callback",
        Some("expected-state"),
        true,
        Duration::from_millis(350),
    )
    .await;
    assert!(res.is_err(), "idle socket must NOT defeat login timeout");
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "expected prompt timeout, elapsed {:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn loopback_callback_ignores_idle_socket_bad_path_and_spoofed_error_probe() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let client_task = tokio::spawn(async move {
        // 1. Keep an idle socket open throughout!
        let _idle = tokio::net::TcpStream::connect(addr).await.expect("idle");
        let client = reqwest::Client::new();
        // 2. Hit `/callbackXYZ` (prefix trick) -> must get 404 and not complete flow
        let r1 = client
            .get(format!("http://{addr}/callbackXYZ?code=evil&state=good-st"))
            .send()
            .await
            .expect("r1");
        assert_eq!(r1.status(), reqwest::StatusCode::NOT_FOUND);
        // 3. Hit `?error=access_denied` with WRONG state -> must get 400 and NOT abort flow
        let r2 = client
            .get(format!(
                "http://{addr}/callback?error=access_denied&state=wrong-st"
            ))
            .send()
            .await
            .expect("r2");
        assert_eq!(r2.status(), reqwest::StatusCode::BAD_REQUEST);
        // 4. Send the real browser callback with valid state -> must succeed immediately!
        let r3 = client
            .get(format!(
                "http://{addr}/callback?code=real-code&state=good-st"
            ))
            .send()
            .await
            .expect("r3");
        assert_eq!(r3.status(), reqwest::StatusCode::OK);
    });

    let (code, st) = wait_for_loopback_callback(
        listener,
        "/callback",
        Some("good-st"),
        true,
        Duration::from_secs(5),
    )
    .await
    .expect("real callback should succeed despite idle socket and spoofed probes");

    assert_eq!(code, "real-code");
    assert_eq!(st.as_deref(), Some("good-st"));
    client_task.await.expect("client task");
}

#[tokio::test]
async fn shared_manager_deduplicates_concurrent_expiry_and_401_refreshes() {
    use std::sync::atomic::AtomicUsize;

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let refresh_count = Arc::new(AtomicUsize::new(0));
    let count_srv = Arc::clone(&refresh_count);

    let server_task = tokio::spawn(async move {
        // Serve up to 2 sequential refresh calls (1 for expiry wave, 1 for 401 wave).
        for wave in 0..2 {
            let (mut s, _) = listener.accept().await.expect("accept");
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).await.expect("read");
            let req = String::from_utf8_lossy(&buf[..n]);
            assert!(req.contains("grant_type=refresh_token"));
            let c = count_srv.fetch_add(1, Ordering::SeqCst);
            // Small delay so the concurrent waiter definitely queues behind the mutex.
            tokio::time::sleep(Duration::from_millis(60)).await;
            let body = if wave == 0 {
                assert!(req.contains("refresh_token=ref-0"));
                format!(
                    r#"{{"access_token":"tok-wave1-{c}","refresh_token":"ref-1","expires_in":3600}}"#
                )
            } else {
                assert!(req.contains("refresh_token=ref-1"));
                format!(
                    r#"{{"access_token":"tok-wave2-{c}","refresh_token":"ref-2","expires_in":3600}}"#
                )
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes()).await.expect("write");
        }
    });

    let dir = std::env::temp_dir().join(format!("carnot-oauth-sf-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));
    let expiring = OAuthStoredToken {
        access_token: "tok-0".into(),
        refresh_token: Some("ref-0".into()),
        token_type: "Bearer".into(),
        expires_at_epoch_secs: Some(now_epoch_secs() + 1),
        scope: None,
        client_id: Some("cid".into()),
        token_url: Some(format!("http://{addr}/token")),
        updated_at_epoch_secs: now_epoch_secs(),
    };
    store.save("sf-prof", expiring.clone()).expect("save");
    let cfg = OAuthConfig {
        flow: OAuthFlow::Pkce,
        token_url: Some(format!("http://{addr}/token")),
        client_id: Some("cid".into()),
        ..Default::default()
    };

    // Two callers resolving via `OAuthTokenManager::shared` share ONE single-flight manager!
    let m1 = OAuthTokenManager::shared(
        "sf-prof",
        expiring.clone(),
        Some(cfg.clone()),
        store.clone(),
    );
    let m2 = OAuthTokenManager::shared("sf-prof", expiring, Some(cfg), store.clone());
    assert!(Arc::ptr_eq(&m1, &m2));

    // Wave 1: two concurrent proactive expiry refreshes -> exactly ONE HTTP refresh!
    let (a, b) = tokio::join!(m1.access_token(None), m2.access_token(None));
    let tok1 = a.expect("a");
    let tok2 = b.expect("b");
    assert_eq!(tok1, tok2);
    assert_eq!(refresh_count.load(Ordering::SeqCst), 1);

    // Wave 2: two concurrent HTTP 401 retries reporting the SAME rejected token -> exactly ONE refresh!
    let (c, d) = tokio::join!(
        m1.access_token(Some(tok1.as_str())),
        m2.access_token(Some(tok1.as_str()))
    );
    let tok3 = c.expect("c");
    let tok4 = d.expect("d");
    assert_eq!(tok3, tok4);
    assert_ne!(tok3, tok1);
    assert_eq!(refresh_count.load(Ordering::SeqCst), 2);

    let on_disk = store.get("sf-prof").expect("disk").expect("present");
    assert_eq!(on_disk.access_token, tok3);
    assert_eq!(on_disk.refresh_token.as_deref(), Some("ref-2"));

    OAuthTokenManager::evict(store.path(), "sf-prof");
    server_task.await.expect("server");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn auth_store_concurrent_saves_and_logout_during_refresh_are_safe() {
    let dir = std::env::temp_dir().join(format!("carnot-oauth-lock-{}", Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));

    std::thread::scope(|scope| {
        for idx in 0..8 {
            let st = store.clone();
            scope.spawn(move || {
                let prof = format!("prof-{idx}");
                save_manual_token(&prof, &format!("tok-{idx}"), &st).expect("save");
            });
        }
    });

    let all = store.list().expect("list");
    assert_eq!(
        all.len(),
        8,
        "file lock must prevent lost writes across concurrent saves"
    );
    for idx in 0..8 {
        assert_eq!(
            all[&format!("prof-{idx}")].access_token,
            format!("tok-{idx}")
        );
    }

    // Removing a profile means `update_if_present` must NOT resurrect it!
    assert!(store.remove("prof-0").expect("remove"));
    let updated = store
        .update_if_present(
            "prof-0",
            OAuthStoredToken {
                access_token: "resurrected".into(),
                refresh_token: None,
                token_type: "Bearer".into(),
                expires_at_epoch_secs: None,
                scope: None,
                client_id: None,
                token_url: None,
                updated_at_epoch_secs: now_epoch_secs(),
            },
        )
        .expect("update_if_present");
    assert!(!updated);
    assert!(store.get("prof-0").expect("get").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}
