//! Pure safe Rust SHA-256, PKCE (`S256`), browser launch, and loopback OAuth callback server.

use super::oauth_err;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use carnot_core::Result;
use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;
use uuid::Uuid;

/// Pure safe Rust FIPS 180-4 SHA-256 (zero unsafe, zero external crates).
#[expect(
    clippy::many_single_char_names,
    reason = "FIPS 180-4 SHA-256 working variables are canonically named a..h"
)]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Computes RFC 7636 PKCE `S256` challenge: `BASE64URL-NO-PAD(SHA256(code_verifier))`.
pub fn pkce_s256_challenge(code_verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(sha256(code_verifier.as_bytes()))
}

/// Generates a CSPRNG PKCE `(code_verifier, code_challenge)` pair (64-char verifier).
pub fn generate_pkce_pair() -> (String, String) {
    let mut entropy = [0u8; 48];
    entropy[0..16].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[16..32].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[32..48].copy_from_slice(Uuid::new_v4().as_bytes());
    let verifier = URL_SAFE_NO_PAD.encode(entropy);
    let challenge = pkce_s256_challenge(&verifier);
    (verifier, challenge)
}

/// Generates a CSPRNG OAuth CSRF `state` token (256-bit entropy).
pub fn generate_oauth_state() -> String {
    let mut entropy = [0u8; 32];
    entropy[0..16].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[16..32].copy_from_slice(Uuid::new_v4().as_bytes());
    URL_SAFE_NO_PAD.encode(entropy)
}

pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Attempts to open a URL in the user's default browser (best-effort; never fatal).
pub fn try_open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
}

const CALLBACK_CONN_READ_TIMEOUT: Duration = Duration::from_secs(5);

async fn handle_loopback_connection(
    mut stream: tokio::net::TcpStream,
    expected_path: String,
    expected_state: Option<String>,
    require_state: bool,
    tx: tokio::sync::mpsc::Sender<Result<(String, Option<String>)>>,
) {
    let mut buf = [0u8; 8192];
    let n = match tokio::time::timeout(CALLBACK_CONN_READ_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => n,
        _ => return,
    };
    let req_str = String::from_utf8_lossy(&buf[..n]);
    let first_line = req_str.lines().next().unwrap_or("");
    let target = first_line.split_whitespace().nth(1).unwrap_or("/");

    let Ok(parsed) = Url::parse(&format!("http://127.0.0.1{target}")) else {
        let _ = write_http_response(&mut stream, "400 Bad Request", "Malformed request.").await;
        return;
    };

    // Exact path match (ignores `/favicon.ico`, `/callbackXYZ`, or stray probes without aborting).
    if parsed.path() != expected_path {
        let _ = write_http_response(&mut stream, "404 Not Found", "Not found").await;
        return;
    }

    let params: BTreeMap<String, String> = parsed.query_pairs().into_owned().collect();

    // Validate CSRF `state` BEFORE checking `error` or `code`, so untrusted local probes cannot
    // abort an active login with `?error=x` or inject a foreign authorization code.
    if let Some(ref exp) = expected_state {
        match params.get("state").map(String::as_str) {
            Some(got) if got == exp.as_str() => {}
            Some(_) => {
                let _ = write_http_response(
                    &mut stream,
                    "400 Bad Request",
                    "OAuth state mismatch. Please retry `carnot auth login`.",
                )
                .await;
                return;
            }
            None if require_state => {
                let _ = write_http_response(
                    &mut stream,
                    "400 Bad Request",
                    "Missing OAuth state parameter. Please retry `carnot auth login`.",
                )
                .await;
                return;
            }
            None => {}
        }
    }

    if let Some(err) = params.get("error") {
        let desc = params
            .get("error_description")
            .map(|d| format!(": {d}"))
            .unwrap_or_default();
        let _ = write_http_response(
            &mut stream,
            "400 Bad Request",
            "Authentication failed. You may close this tab and return to Carnot.",
        )
        .await;
        let _ = tx
            .send(Err(oauth_err(
                "oauth",
                format!("OAuth provider returned error `{err}`{desc}"),
            )))
            .await;
        return;
    }

    let Some(code) = params.get("code").filter(|s| !s.is_empty()) else {
        let _ = write_http_response(
            &mut stream,
            "400 Bad Request",
            "Missing authorization code in callback.",
        )
        .await;
        return;
    };

    let _ = write_http_response(
        &mut stream,
        "200 OK",
        "Carnot authentication succeeded. You may close this tab and return to your terminal.",
    )
    .await;
    let _ = tx
        .send(Ok((code.clone(), params.get("state").cloned())))
        .await;
}

pub(super) async fn wait_for_loopback_callback(
    listener: TcpListener,
    expected_path: &str,
    expected_state: Option<&str>,
    require_state: bool,
    timeout: Duration,
) -> Result<(String, Option<String>)> {
    let deadline = tokio::time::Instant::now() + timeout;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<(String, Option<String>)>>(8);

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(oauth_err(
                "oauth",
                "OAuth login timed out waiting for browser callback".into(),
            ));
        }

        tokio::select! {
            res = rx.recv() => {
                if let Some(outcome) = res {
                    return outcome;
                }
            }
            accepted = tokio::time::timeout(remaining, listener.accept()) => {
                let (stream, _) = accepted
                    .map_err(|_| {
                        oauth_err(
                            "oauth",
                            "OAuth login timed out waiting for browser callback".into(),
                        )
                    })?
                    .map_err(|e| oauth_err("oauth", format!("OAuth callback accept failed: {e}")))?;

                let path = expected_path.to_string();
                let state = expected_state.map(ToString::to_string);
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    handle_loopback_connection(stream, path, state, require_state, tx_clone).await;
                });
            }
        }
    }
}

async fn write_http_response(
    stream: &mut tokio::net::TcpStream,
    status: &str,
    message: &str,
) -> std::io::Result<()> {
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Carnot OAuth</title></head>\
         <body style=\"font-family:system-ui,sans-serif;max-width:480px;margin:4rem auto;padding:1rem;\">\
         <h2>Carnot</h2><p>{message}</p></body></html>"
    );
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
        html.len()
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await
}

/// Extracts an authorization `code` from either a full pasted callback URL or a bare code string.
pub fn extract_code_from_manual_input(input: &str, expected_state: Option<&str>) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(oauth_err(
            "oauth",
            "empty authorization code / callback URL".into(),
        ));
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let parsed = Url::parse(trimmed)
            .map_err(|e| oauth_err("oauth", format!("invalid pasted callback URL: {e}")))?;
        let params: BTreeMap<String, String> = parsed.query_pairs().into_owned().collect();
        if let Some(exp) = expected_state
            && params.get("state").map(String::as_str) != Some(exp)
        {
            return Err(oauth_err(
                "oauth",
                "OAuth callback CSRF `state` mismatch".into(),
            ));
        }
        if let Some(err) = params.get("error") {
            return Err(oauth_err(
                "oauth",
                format!("OAuth callback contained error `{err}`"),
            ));
        }
        return params
            .get("code")
            .filter(|s| !s.is_empty())
            .cloned()
            .ok_or_else(|| {
                oauth_err(
                    "oauth",
                    "pasted callback URL is missing `code` query parameter".into(),
                )
            });
    }
    Ok(trimmed.to_string())
}

