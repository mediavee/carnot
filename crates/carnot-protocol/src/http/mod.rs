//! Hardened embedded HTTP/1.1 + SSE API server powered by `hyper`.
//!
//! Security & concurrency invariants:
//!   - Bearer token authentication (`CARNOT_HTTP_TOKEN` or auto-generated 256-bit token saved in
//!     `~/.carnot/server.token` with mode `0600`) required on all `/v1/*` routes.
//!   - Strict `Host` & `Origin` localhost checks + `Content-Type: application/json` enforcement on
//!     POST routes with bodies (blocks browser CSRF & DNS rebinding).
//!   - Bounded header count (64), header buffer size (16 KiB), body size (8 MiB), and 15 s header
//!     & body read timeouts — enforced before allocating/reading unauthenticated request bodies.
//!   - Full RFC 9112 HTTP/1.1 conformance via `hyper` (including `Transfer-Encoding: chunked` and
//!     persistent keep-alive connections).
//!   - Returns `409 Conflict` if a second `POST /v1/prompt` arrives while a run is active.
//!   - Lossless SSE streaming from `EventLog` with `id: <seq>` & `Last-Event-ID` replay on `GET /v1/events`.

use crate::interaction::ChannelInteractionBroker;
use bytes::Bytes;
use carnot_agent::Agent;
use carnot_core::{AgentEvent, CacheTtl, ContentBlock, Result, UserQuestionAnswer};
use futures_util::Stream;
use http::header::{
    AUTHORIZATION, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE, HOST, ORIGIN, TRANSFER_ENCODING,
};
use http::{Method, Request, Response, StatusCode};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, Limited, StreamBody};
use hyper::body::{Body as _, Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

const MAX_HEADERS: usize = 64;
const MAX_HEADER_BUF_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_CONCURRENT_CONNECTIONS: usize = 128;
const REQUEST_READ_TIMEOUT_SECS: u64 = 15;
const SSE_KEEPALIVE_SECS: u64 = 30;

type HttpBody = BoxBody<Bytes, Infallible>;
type HttpResponse = Response<HttpBody>;

struct SseFrameStream {
    rx: mpsc::Receiver<Bytes>,
}

impl Stream for SseFrameStream {
    type Item = std::result::Result<Frame<Bytes>, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx
            .poll_recv(cx)
            .map(|opt| opt.map(|chunk| Ok(Frame::data(chunk))))
    }
}

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
                if let Err(e) = write_token_0600(&tok_path, &t) {
                    tracing::warn!(
                        token_file = %tok_path.display(),
                        error = %e,
                        "failed to write carnot HTTP Bearer token file"
                    );
                } else {
                    tracing::info!(
                        token_file = %tok_path.display(),
                        "carnot HTTP Bearer token written (mode 0600)"
                    );
                }
            }
            t
        }
    };
    serve_http_with_token(agent, addr, token).await
}

pub async fn serve_http_with_token(agent: Arc<Agent>, addr: &str, token: String) -> Result<()> {
    let broker = ChannelInteractionBroker::new(Arc::clone(&agent));
    agent
        .set_question_broker(Arc::clone(&broker) as Arc<dyn carnot_ext::UserQuestionBroker>)
        .await;

    let listener = TcpListener::bind(addr).await?;
    let expected_token: Arc<str> = Arc::from(token);
    let conn_limit = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
    let prompt_gate = Arc::new(tokio::sync::Mutex::new(()));
    tracing::info!(
        %addr,
        "carnot HTTP/SSE server listening (hyper HTTP/1.1 + Bearer auth + CSRF protection)"
    );
    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                let (stream, _) = match accept_res {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::warn!(error = %e, "HTTP listener accept error; continuing");
                        tokio::time::sleep(Duration::from_millis(25)).await;
                        continue;
                    }
                };
                let Ok(permit) = Arc::clone(&conn_limit).try_acquire_owned() else {
                    drop(stream);
                    continue;
                };
                let ag = Arc::clone(&agent);
                let br = Arc::clone(&broker);
                let tok = Arc::clone(&expected_token);
                let pgate = Arc::clone(&prompt_gate);
                tokio::spawn(async move {
                    let _permit = permit;
                    serve_conn(ag, br, pgate, stream, tok).await;
                });
            }
            () = shutdown_signal() => {
                tracing::info!("carnot HTTP/SSE server shutting down gracefully");
                agent.abort().await;
                break;
            }
        }
    }
    Ok(())
}

async fn serve_conn(
    agent: Arc<Agent>,
    broker: Arc<ChannelInteractionBroker>,
    prompt_gate: Arc<tokio::sync::Mutex<()>>,
    stream: TcpStream,
    expected_token: Arc<str>,
) {
    let io = TokioIo::new(stream);
    let service = service_fn(move |req: Request<Incoming>| {
        let ag = Arc::clone(&agent);
        let br = Arc::clone(&broker);
        let pg = Arc::clone(&prompt_gate);
        let tok = Arc::clone(&expected_token);
        async move { Ok::<_, Infallible>(handle_request(ag, br, pg, req, &tok).await) }
    });
    let mut builder = http1::Builder::new();
    builder
        .max_headers(MAX_HEADERS)
        .max_buf_size(MAX_HEADER_BUF_BYTES)
        .timer(TokioTimer::new())
        .header_read_timeout(Duration::from_secs(REQUEST_READ_TIMEOUT_SECS))
        .half_close(true)
        .keep_alive(true);
    let _ = builder.serve_connection(io, service).await;
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            () = async {
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

fn is_valid_optional_port(rest: &str) -> bool {
    if rest.is_empty() {
        return true;
    }
    rest.strip_prefix(':').is_some_and(|p| {
        !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok()
    })
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim();
    if h.is_empty()
        || h.bytes().any(|b| {
            b.is_ascii_whitespace()
                || b.is_ascii_control()
                || matches!(b, b'@' | b'/' | b'?' | b'#')
        })
    {
        return false;
    }
    let lower = h.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("[::1]") {
        return is_valid_optional_port(rest);
    }
    if let Some(rest) = lower.strip_prefix("127.0.0.1") {
        return is_valid_optional_port(rest);
    }
    if let Some(rest) = lower.strip_prefix("localhost") {
        return is_valid_optional_port(rest);
    }
    false
}

fn is_allowed_origin(origin: &str) -> bool {
    let o = origin.trim();
    let Some(authority) = o
        .strip_prefix("http://")
        .or_else(|| o.strip_prefix("https://"))
    else {
        return false;
    };
    is_loopback_host(authority)
}

fn header_str<'a>(req: &'a Request<Incoming>, name: &http::HeaderName) -> Option<&'a str> {
    req.headers().get(name).and_then(|v| v.to_str().ok())
}

fn json_response(status: StatusCode, body: &Value) -> HttpResponse {
    let payload = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(payload)).boxed())
        .unwrap_or_else(|_| Response::new(Full::new(Bytes::from_static(b"{}")).boxed()))
}

fn sse_response(rx: mpsc::Receiver<Bytes>) -> HttpResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .body(StreamBody::new(SseFrameStream { rx }).boxed())
        .unwrap_or_else(|_| Response::new(Full::new(Bytes::new()).boxed()))
}

async fn handle_request(
    agent: Arc<Agent>,
    broker: Arc<ChannelInteractionBroker>,
    prompt_gate: Arc<tokio::sync::Mutex<()>>,
    req: Request<Incoming>,
    expected_token: &str,
) -> HttpResponse {
    // Host & Origin anti-CSRF / anti-DNS-rebinding guards (fail-closed: Host is required in HTTP/1.1).
    match header_str(&req, &HOST) {
        Some(h) if is_loopback_host(h) => {}
        Some(_) => {
            return json_response(StatusCode::FORBIDDEN, &json!({"error": "forbidden host"}));
        }
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &json!({"error": "missing Host header"}),
            );
        }
    }
    if let Some(o) = header_str(&req, &ORIGIN)
        && !is_allowed_origin(o)
    {
        return json_response(StatusCode::FORBIDDEN, &json!({"error": "forbidden origin"}));
    }

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if method == Method::GET && path == "/health" {
        return json_response(
            StatusCode::OK,
            &json!({"status": "ok", "engine": "carnot", "version": env!("CARGO_PKG_VERSION")}),
        );
    }

    // Require Bearer token on all `/v1/*` routes BEFORE allocating or reading any request body.
    let authorized = header_str(&req, &AUTHORIZATION)
        .and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("bearer "))
        })
        .is_some_and(|tok| constant_time_eq(tok.trim().as_bytes(), expected_token.as_bytes()));
    if !authorized {
        return json_response(StatusCode::UNAUTHORIZED, &json!({"error": "unauthorized"}));
    }

    // Reject declared payloads exceeding 8 MiB before reading body bytes.
    if let Some(cl_val) = header_str(&req, &CONTENT_LENGTH) {
        match cl_val.trim().parse::<usize>() {
            Ok(n) if n > MAX_BODY_BYTES => {
                return json_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    &json!({"error": "payload exceeds 8 MiB limit"}),
                );
            }
            Ok(_) => {}
            Err(_) => {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": "invalid Content-Length header"}),
                );
            }
        }
    }
    let has_body = req.headers().contains_key(TRANSFER_ENCODING)
        || req.body().size_hint().lower() > 0
        || req
            .body()
            .size_hint()
            .upper()
            .is_some_and(|upper| upper > 0);

    // Enforce `application/json` on POST requests with a body (blocks `fetch(..., {mode:"no-cors"})` CSRF!).
    if method == Method::POST && has_body {
        let is_json = header_str(&req, &CONTENT_TYPE)
            .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("application/json"));
        if !is_json {
            return json_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                &json!({"error": "Content-Type must be application/json"}),
            );
        }
    }

    let last_event_id: Option<u64> = req
        .headers()
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());

    let body_bytes = if has_body {
        let limited = Limited::new(req.into_body(), MAX_BODY_BYTES);
        match tokio::time::timeout(
            Duration::from_secs(REQUEST_READ_TIMEOUT_SECS),
            limited.collect(),
        )
        .await
        {
            Ok(Ok(collected)) => collected.to_bytes(),
            Ok(Err(err)) => {
                if err.is::<http_body_util::LengthLimitError>() {
                    return json_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &json!({"error": "payload exceeds 8 MiB limit"}),
                    );
                }
                return json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": format!("failed to read request body: {err}")}),
                );
            }
            Err(_) => {
                return json_response(
                    StatusCode::REQUEST_TIMEOUT,
                    &json!({"error": "request body timeout"}),
                );
            }
        }
    } else {
        Bytes::new()
    };

    match (method, path.as_str()) {
        (Method::GET, "/v1/metrics") => {
            let met = agent.metrics().await;
            let sid = agent.session_id().await;
            let ph = agent.prefix_hash().await;
            let seq = agent.latest_event_seq();
            let (ctx_tokens, ctx_breakdown) = agent.current_context_with_breakdown().await;
            let ctx_window = agent.current_context_window().await;
            let ttl = agent.current_cache_ttl().await;
            json_response(
                StatusCode::OK,
                &json!({
                    "session_id": sid,
                    "prefix_hash": ph,
                    "latest_seq": seq,
                    "cache_ttl": ttl.as_str(),
                    "cache_hit_rate": met.cache_hit_rate(),
                    "prefix_retention_rate": met.prefix_retention_rate(),
                    "context_tokens": ctx_tokens,
                    "context_window": ctx_window,
                    "context_breakdown": ctx_breakdown,
                    "metrics": met,
                }),
            )
        }
        (Method::POST, "/v1/cache-ttl") => {
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let raw = payload
                .get("ttl")
                .or_else(|| payload.get("cache_ttl"))
                .or_else(|| payload.get("retention"))
                .and_then(Value::as_str)
                .unwrap_or("");
            match CacheTtl::parse(raw) {
                Some(ttl) => {
                    agent.set_cache_ttl(ttl).await;
                    json_response(StatusCode::OK, &json!({"cache_ttl": ttl.as_str()}))
                }
                None => json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": format!("invalid cache_ttl `{raw}` (expected short|hybrid|long)")}),
                ),
            }
        }
        (Method::GET, "/v1/goal") => match agent.goal_controller() {
            Some(ctrl) => json_response(StatusCode::OK, &ctrl.status_json()),
            None => json_response(
                StatusCode::NOT_IMPLEMENTED,
                &json!({"error": "goal extension not enabled on this agent"}),
            ),
        },
        (Method::POST, "/v1/goal") => match agent.goal_controller() {
            Some(ctrl) => {
                let payload: Value = match serde_json::from_slice::<Value>(&body_bytes) {
                    Ok(v) if v.is_object() => v,
                    Ok(_) => {
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": "POST /v1/goal body must be a JSON object"}),
                        );
                    }
                    Err(e) => {
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": format!("invalid JSON body: {e}")}),
                        );
                    }
                };
                let criterion = payload
                    .get("criterion")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let verify_commands: Vec<String> = payload
                    .get("verify_commands")
                    .or_else(|| payload.get("commands"))
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                let max_iterations = payload
                    .get("max_iterations")
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok());
                let verifier_model = payload
                    .get("verifier_model")
                    .or_else(|| payload.get("model"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let st = ctrl.set_goal(criterion, verify_commands, max_iterations, verifier_model);
                json_response(StatusCode::OK, &st)
            }
            None => json_response(
                StatusCode::NOT_IMPLEMENTED,
                &json!({"error": "goal extension not enabled on this agent"}),
            ),
        },
        (Method::DELETE, "/v1/goal") => match agent.goal_controller() {
            Some(ctrl) => json_response(StatusCode::OK, &ctrl.clear_goal()),
            None => json_response(
                StatusCode::NOT_IMPLEMENTED,
                &json!({"error": "goal extension not enabled on this agent"}),
            ),
        },
        (Method::GET, "/v1/events") => {
            let (tx, rx) = mpsc::channel::<Bytes>(64);
            let mut ev_stream = agent.subscribe_from(last_event_id.unwrap_or(0));
            tokio::spawn(async move {
                loop {
                    if let Ok(env) = tokio::time::timeout(
                        Duration::from_secs(SSE_KEEPALIVE_SECS),
                        ev_stream.next(),
                    )
                    .await
                    {
                        if let Ok(line) = serde_json::to_string(&env) {
                            let frame = format!("id: {}\ndata: {line}\n\n", env.seq);
                            if tx.send(Bytes::from(frame)).await.is_err() {
                                break;
                            }
                        }
                    } else if tx
                        .send(Bytes::from_static(b": keepalive\n\n"))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
            sse_response(rx)
        }
        (Method::POST, "/v1/steer") => {
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let msg = payload
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            agent.steer(msg).await;
            json_response(StatusCode::OK, &json!({"queued": true}))
        }
        (Method::POST, "/v1/follow-up") => {
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let msg = payload
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            agent.follow_up(msg).await;
            json_response(StatusCode::OK, &json!({"queued": true}))
        }
        (Method::GET, "/v1/pending") => {
            let pending = broker.list_pending().await;
            json_response(StatusCode::OK, &pending)
        }
        (Method::POST, "/v1/permissions") => {
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let req_id = payload
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let decision = payload
                .get("decision")
                .and_then(Value::as_str)
                .unwrap_or("deny");
            let reason = payload
                .get("reason")
                .and_then(Value::as_str)
                .map(str::to_string);
            let resolved = broker.respond_permission(req_id, decision, reason).await;
            let status = if resolved {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            };
            json_response(status, &json!({"resolved": resolved}))
        }
        (Method::POST, "/v1/questions") => {
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let req_id = payload
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let answers: Vec<UserQuestionAnswer> = payload
                .get("answers")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            let resolved = broker.respond_question(req_id, answers).await;
            let status = if resolved {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            };
            json_response(status, &json!({"resolved": resolved}))
        }
        (Method::POST, "/v1/abort") => {
            agent.abort().await;
            json_response(StatusCode::OK, &json!({"aborted": true}))
        }
        (Method::POST, "/v1/compact") => {
            if agent.is_busy() {
                return json_response(
                    StatusCode::CONFLICT,
                    &json!({"error": "agent run already in progress; cancel before compacting"}),
                );
            }
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let focus = payload
                .get("focus")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty());
            match agent.compact_now_with_focus(focus).await {
                Ok((before, after, semantic)) => json_response(
                    StatusCode::OK,
                    &json!({
                        "tokens_before": before,
                        "tokens_after": after,
                        "semantic_llm": semantic,
                    }),
                ),
                Err(e) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &json!({"error": e.to_string()}),
                ),
            }
        }
        (Method::POST, "/v1/undo") => match agent.undo_last_edit() {
            Ok(path) => json_response(
                StatusCode::OK,
                &json!({"reverted": path.map(|p| p.display().to_string())}),
            ),
            Err(e) => json_response(StatusCode::CONFLICT, &json!({"error": e.to_string()})),
        },
        (Method::POST, "/v1/prompt") => {
            let Ok(prompt_guard) = prompt_gate.try_lock_owned() else {
                return json_response(
                    StatusCode::CONFLICT,
                    &json!({"error": "agent run already in progress; use /v1/steer or /v1/abort"}),
                );
            };
            if agent.is_busy() {
                drop(prompt_guard);
                return json_response(
                    StatusCode::CONFLICT,
                    &json!({"error": "agent run already in progress; use /v1/steer or /v1/abort"}),
                );
            }
            let payload: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let prompt = payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let mut image_blocks = Vec::new();
            if let Some(images) = payload.get("images").and_then(Value::as_array) {
                for img in images {
                    let media_type = img
                        .get("media_type")
                        .or_else(|| img.get("mimeType"))
                        .and_then(Value::as_str)
                        .unwrap_or("image/png");
                    if let Some(data) = img
                        .get("data_base64")
                        .or_else(|| img.get("data"))
                        .and_then(Value::as_str)
                        && !data.is_empty()
                    {
                        image_blocks.push(ContentBlock::Image {
                            media_type: media_type.to_string(),
                            data_base64: data.to_string(),
                        });
                    }
                }
            }
            let stream_sse = payload
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            if stream_sse {
                let (tx, rx) = mpsc::channel::<Bytes>(64);
                let start_seq = agent.latest_event_seq();
                let mut ev_stream = agent.subscribe_from(start_seq);
                let ag_run = Arc::clone(&agent);
                let run_task = tokio::spawn(async move {
                    let _prompt_guard = prompt_guard;
                    if image_blocks.is_empty() {
                        ag_run.prompt(prompt).await
                    } else {
                        let mut blocks = Vec::with_capacity(1 + image_blocks.len());
                        if !prompt.is_empty() {
                            blocks.push(ContentBlock::Text { text: prompt });
                        }
                        blocks.extend(image_blocks);
                        ag_run.prompt_blocks(blocks).await
                    }
                });
                tokio::spawn(async move {
                    loop {
                        if let Ok(env) = tokio::time::timeout(
                            Duration::from_secs(SSE_KEEPALIVE_SECS),
                            ev_stream.next(),
                        )
                        .await
                        {
                            let is_done = matches!(
                                env.event,
                                AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                            );
                            if let Ok(line) = serde_json::to_string(&env) {
                                let frame = format!("id: {}\ndata: {line}\n\n", env.seq);
                                if tx.send(Bytes::from(frame)).await.is_err() {
                                    break;
                                }
                            }
                            if is_done {
                                break;
                            }
                        } else if tx
                            .send(Bytes::from_static(b": keepalive\n\n"))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    let _ = run_task.await;
                });
                sse_response(rx)
            } else {
                let res = if image_blocks.is_empty() {
                    agent.prompt(prompt).await
                } else {
                    let mut blocks = Vec::with_capacity(1 + image_blocks.len());
                    if !prompt.is_empty() {
                        blocks.push(ContentBlock::Text { text: prompt });
                    }
                    blocks.extend(image_blocks);
                    agent.prompt_blocks(blocks).await
                };
                match res {
                    Ok(o) => json_response(
                        StatusCode::OK,
                        &json!({
                            "final_text": o.final_text,
                            "metrics": o.metrics,
                            "halted_reason": o.halted_reason,
                        }),
                    ),
                    Err(e) => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &json!({"error": e.to_string()}),
                    ),
                }
            }
        }
        _ => json_response(StatusCode::NOT_FOUND, &json!({"error": "not found"})),
    }
}

#[cfg(test)]
mod tests;
