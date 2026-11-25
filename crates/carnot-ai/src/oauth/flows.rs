//! Standards-compliant OAuth 2.0 flows: PKCE, PKCE key-exchange, Device Code, and Refresh.

use super::pkce::{
    extract_code_from_manual_input, generate_oauth_state, generate_pkce_pair, now_epoch_secs,
    try_open_browser, wait_for_loopback_callback,
};
use super::store::AuthStore;
use super::{
    OAuthConfig, OAuthFlow, OAuthLoginOptions, OAuthPrompt, OAuthStoredToken, encode_form,
    oauth_err,
};
use crate::build_http_client_for_url;
use carnot_core::{CarnotError, Result};
use serde_json::Value;
use std::time::Duration;
use tokio::net::TcpListener;
use url::Url;

/// Runs a complete OAuth login flow for `profile` according to `cfg` and persists the resulting token.
pub async fn login_with_oauth<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    store: &AuthStore,
    opts: &OAuthLoginOptions,
    mut on_prompt: F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let flow = if opts.force_device_flow {
        OAuthFlow::DeviceCode
    } else {
        cfg.flow
    };

    let token = match flow {
        OAuthFlow::PkceKeyExchange => {
            run_pkce_key_exchange_flow(profile, cfg, opts, &mut on_prompt, read_manual_input)
                .await?
        }
        OAuthFlow::Pkce => {
            run_standard_pkce_flow(profile, cfg, opts, &mut on_prompt, read_manual_input).await?
        }
        OAuthFlow::DeviceCode => run_device_code_flow(profile, cfg, opts, &mut on_prompt).await?,
    };

    store.save(profile, token.clone())?;
    Ok(token)
}

/// Saves a manually supplied OAuth access token or API key into the `0600` auth store.
pub fn save_manual_token(
    profile: &str,
    access_token: &str,
    store: &AuthStore,
) -> Result<OAuthStoredToken> {
    let trimmed = access_token.trim();
    if trimmed.is_empty() {
        return Err(oauth_err("oauth", "empty token provided".into()));
    }
    let tok = OAuthStoredToken {
        access_token: trimmed.to_string(),
        refresh_token: None,
        token_type: "Bearer".into(),
        expires_at_epoch_secs: None,
        scope: None,
        client_id: None,
        token_url: None,
        updated_at_epoch_secs: now_epoch_secs(),
    };
    store.save(profile, tok.clone())?;
    Ok(tok)
}

/// Forces an immediate refresh of the stored token for `profile` and saves the updated credential.
pub async fn refresh_stored_token(
    profile: &str,
    oauth_cfg: Option<&OAuthConfig>,
    store: &AuthStore,
) -> Result<OAuthStoredToken> {
    let current = store.get(profile)?.ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("no stored OAuth token for `{profile}`; run `carnot auth login {profile}`"),
        )
    })?;
    let refreshed = exchange_refresh_token(profile, &current, oauth_cfg).await?;
    store.save(profile, refreshed.clone())?;
    Ok(refreshed)
}

async fn run_pkce_key_exchange_flow<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let authorize_base = cfg.authorize_url.as_deref().ok_or_else(|| {
        oauth_err(
            profile,
            format!("provider `{profile}` OAuth config is missing `authorize_url`"),
        )
    })?;
    let exchange_url = cfg.exchange_url.as_deref().ok_or_else(|| {
        oauth_err(
            profile,
            format!("provider `{profile}` OAuth config is missing `exchange_url`"),
        )
    })?;
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_oauth_state();
    let callback_path = cfg.callback_path();

    let (listener_opt, callback_url) = if opts.manual_callback {
        // Per OpenRouter PKCE docs, omitting `callback_url` in headless/manual mode makes the
        // authorization page display the code directly for manual paste.
        (None, None)
    } else {
        let bind_port = opts.port_override.or(cfg.redirect_port).unwrap_or(0);
        let listener = TcpListener::bind(("127.0.0.1", bind_port))
            .await
            .map_err(|e| {
                oauth_err(
                    "oauth",
                    format!(
                        "failed to bind local OAuth callback listener on 127.0.0.1:{bind_port}: {e}"
                    ),
                )
            })?;
        let port = listener
            .local_addr()
            .map_err(|e| oauth_err("oauth", format!("local_addr failed: {e}")))?
            .port();
        (
            Some(listener),
            Some(format!("http://localhost:{port}{callback_path}")),
        )
    };

    let mut auth_url = Url::parse(authorize_base).map_err(|e| {
        oauth_err(
            "oauth",
            format!("invalid OAuth authorize_url `{authorize_base}`: {e}"),
        )
    })?;
    {
        let mut qp = auth_url.query_pairs_mut();
        if let Some(ref cb) = callback_url {
            qp.append_pair("callback_url", cb);
        }
        qp.append_pair("code_challenge", &code_challenge);
        qp.append_pair("code_challenge_method", "S256");
        qp.append_pair("state", &state);
        for (k, v) in &cfg.extra_auth_params {
            qp.append_pair(k, v);
        }
    }

    let auth_url_str = auth_url.to_string();
    on_prompt(OAuthPrompt::OpenUrl {
        url: auth_url_str.clone(),
        callback_url: callback_url.unwrap_or_default(),
        manual_paste: opts.manual_callback,
    });
    if opts.open_browser && !opts.manual_callback {
        try_open_browser(&auth_url_str);
    }

    let code = if let Some(listener) = listener_opt {
        let (code, _state) =
            wait_for_loopback_callback(listener, callback_path, Some(&state), false, opts.timeout)
                .await?;
        code
    } else {
        let pasted = read_manual_input()?;
        extract_code_from_manual_input(&pasted, None)?
    };

    let client = build_http_client_for_url(exchange_url);
    let resp = client
        .post(exchange_url)
        .json(&serde_json::json!({
            "code": code,
            "code_verifier": code_verifier,
            "code_challenge_method": "S256",
        }))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth key exchange request failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("PKCE key exchange HTTP {status}: {text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
    let body: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid JSON from PKCE key exchange: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;
    let key = body
        .get("key")
        .or_else(|| body.get("access_token"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "PKCE key-exchange response missing `key`".into(),
            retryable: false,
            retry_after_ms: None,
        })?;

    Ok(OAuthStoredToken {
        access_token: key.to_string(),
        refresh_token: None,
        token_type: "Bearer".into(),
        expires_at_epoch_secs: None,
        scope: None,
        client_id: None,
        token_url: None,
        updated_at_epoch_secs: now_epoch_secs(),
    })
}

async fn run_standard_pkce_flow<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let authorize_base = cfg.authorize_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `authorize_url`"),
        )
    })?;
    let token_url = cfg.token_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `token_url`"),
        )
    })?;
    let client_id = cfg
        .resolve_client_id(opts.client_id_override.as_deref())
        .ok_or_else(|| {
            let hint = cfg.client_id_env.as_deref().map_or_else(
                || {
                    " (pass `--client-id` or set `oauth.client_id` in ~/.carnot/providers.json)"
                        .into()
                },
                |e| format!(" (set `{e}` or pass `--client-id`)"),
            );
            oauth_err(
                "oauth",
                format!(
                    "provider `{profile}` OAuth login requires a registered OAuth `client_id`{hint}"
                ),
            )
        })?;
    let client_secret = cfg.resolve_client_secret();
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_oauth_state();
    let callback_path = cfg.callback_path();

    let (listener_opt, redirect_uri) =
        if opts.manual_callback {
            let port = opts.port_override.or(cfg.redirect_port).unwrap_or(8787);
            (None, format!("http://127.0.0.1:{port}{callback_path}"))
        } else {
            let bind_port = opts.port_override.or(cfg.redirect_port).unwrap_or(0);
            let listener = TcpListener::bind(("127.0.0.1", bind_port))
            .await
            .map_err(|e| {
                oauth_err("oauth", format!(
                    "failed to bind local OAuth callback listener on 127.0.0.1:{bind_port}: {e}"
                ))
            })?;
            let port = listener
                .local_addr()
                .map_err(|e| oauth_err("oauth", format!("local_addr failed: {e}")))?
                .port();
            (
                Some(listener),
                format!("http://127.0.0.1:{port}{callback_path}"),
            )
        };

    let mut auth_url = Url::parse(authorize_base).map_err(|e| {
        oauth_err(
            "oauth",
            format!("invalid OAuth authorize_url `{authorize_base}`: {e}"),
        )
    })?;
    {
        let mut qp = auth_url.query_pairs_mut();
        qp.append_pair("response_type", "code");
        qp.append_pair("client_id", &client_id);
        qp.append_pair("redirect_uri", &redirect_uri);
        qp.append_pair("code_challenge", &code_challenge);
        qp.append_pair("code_challenge_method", "S256");
        qp.append_pair("state", &state);
        if !cfg.scopes.is_empty() {
            qp.append_pair("scope", &cfg.scopes.join(" "));
        }
        for (k, v) in &cfg.extra_auth_params {
            qp.append_pair(k, v);
        }
    }

    let auth_url_str = auth_url.to_string();
    on_prompt(OAuthPrompt::OpenUrl {
        url: auth_url_str.clone(),
        callback_url: redirect_uri.clone(),
        manual_paste: opts.manual_callback,
    });
    if opts.open_browser && !opts.manual_callback {
        try_open_browser(&auth_url_str);
    }

    let code = if let Some(listener) = listener_opt {
        let (code, _) =
            wait_for_loopback_callback(listener, callback_path, Some(&state), true, opts.timeout)
                .await?;
        code
    } else {
        let pasted = read_manual_input()?;
        extract_code_from_manual_input(&pasted, Some(&state))?
    };

    let mut form: Vec<(String, String)> = vec![
        ("grant_type".into(), "authorization_code".into()),
        ("code".into(), code),
        ("redirect_uri".into(), redirect_uri),
        ("client_id".into(), client_id.clone()),
        ("code_verifier".into(), code_verifier),
    ];
    if let Some(secret) = client_secret {
        form.push(("client_secret".into(), secret));
    }
    for (k, v) in &cfg.extra_token_params {
        form.push((k.clone(), v.clone()));
    }

    let client = build_http_client_for_url(token_url);
    let resp = client
        .post(token_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token exchange failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    parse_token_response(profile, resp, Some(&client_id), Some(token_url), None).await
}

async fn run_device_code_flow<F>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
{
    let device_url = cfg.device_authorization_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `device_authorization_url`"),
        )
    })?;
    let token_url = cfg.token_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `token_url`"),
        )
    })?;
    let client_id = cfg
        .resolve_client_id(opts.client_id_override.as_deref())
        .ok_or_else(|| {
            oauth_err(
                "oauth",
                format!("provider `{profile}` device-code OAuth requires a `client_id`"),
            )
        })?;
    let client_secret = cfg.resolve_client_secret();

    let mut form: Vec<(String, String)> = vec![("client_id".into(), client_id.clone())];
    if !cfg.scopes.is_empty() {
        form.push(("scope".into(), cfg.scopes.join(" ")));
    }
    for (k, v) in &cfg.extra_auth_params {
        form.push((k.clone(), v.clone()));
    }

    let client = build_http_client_for_url(device_url);
    let resp = client
        .post(device_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("device authorization request failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    let status = resp.status();
    let body_text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("device authorization HTTP {status}: {body_text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
    let body: Value = serde_json::from_str(&body_text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid device authorization JSON: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;

    let device_code = body
        .get("device_code")
        .and_then(Value::as_str)
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `device_code`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let user_code = body
        .get("user_code")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `user_code`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let verification_uri = body
        .get("verification_uri")
        .or_else(|| body.get("verification_url"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `verification_uri`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let verification_uri_complete = body
        .get("verification_uri_complete")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_u64)
        .unwrap_or(600);
    let mut interval_secs = body
        .get("interval")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .max(1);

    on_prompt(OAuthPrompt::DeviceCode {
        verification_uri: verification_uri.clone(),
        verification_uri_complete: verification_uri_complete.clone(),
        user_code,
    });
    if opts.open_browser {
        if let Some(ref full) = verification_uri_complete {
            try_open_browser(full);
        } else if !verification_uri.is_empty() {
            try_open_browser(&verification_uri);
        }
    }

    let deadline =
        tokio::time::Instant::now() + opts.timeout.min(Duration::from_secs(expires_in.max(10)));
    let token_client = build_http_client_for_url(token_url);
    let mut consecutive_net_errs: u32 = 0;

    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(CarnotError::Provider {
                provider: profile.into(),
                message: "device authorization timed out before approval".into(),
                retryable: false,
                retry_after_ms: None,
            });
        }
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;

        let mut poll_form: Vec<(String, String)> = vec![
            (
                "grant_type".into(),
                "urn:ietf:params:oauth:grant-type:device_code".into(),
            ),
            ("device_code".into(), device_code.clone()),
            ("client_id".into(), client_id.clone()),
        ];
        if let Some(ref secret) = client_secret {
            poll_form.push(("client_secret".into(), secret.clone()));
        }
        for (k, v) in &cfg.extra_token_params {
            poll_form.push((k.clone(), v.clone()));
        }

        let r = match token_client
            .post(token_url)
            .header("accept", "application/json")
            .header("content-type", "application/x-www-form-urlencoded")
            .body(encode_form(&poll_form))
            .send()
            .await
        {
            Ok(resp) => {
                consecutive_net_errs = 0;
                resp
            }
            Err(e) => {
                consecutive_net_errs += 1;
                if consecutive_net_errs >= 5 {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: format!(
                            "device token poll failed after {consecutive_net_errs} consecutive network errors: {}",
                            e.without_url()
                        ),
                        retryable: true,
                        retry_after_ms: None,
                    });
                }
                tracing::debug!(
                    provider = %profile,
                    attempt = consecutive_net_errs,
                    "transient network error during device-code poll; retrying"
                );
                continue;
            }
        };

        let status = r.status();
        let text = r.text().await.unwrap_or_default();
        if status.is_success() {
            let v: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
                provider: profile.into(),
                message: format!("invalid token JSON: {e}"),
                retryable: false,
                retry_after_ms: None,
            })?;
            if let Some(err_code) = v.get("error").and_then(Value::as_str) {
                match err_code {
                    "authorization_pending" => continue,
                    "slow_down" => {
                        interval_secs = interval_secs.saturating_add(5);
                        continue;
                    }
                    other => {
                        return Err(CarnotError::Provider {
                            provider: profile.into(),
                            message: format!("device authorization failed: {other}"),
                            retryable: false,
                            retry_after_ms: None,
                        });
                    }
                }
            }
            return parse_token_value(profile, &v, Some(&client_id), Some(token_url), None);
        }

        if let Ok(v) = serde_json::from_str::<Value>(&text)
            && let Some(err_code) = v.get("error").and_then(Value::as_str)
        {
            match err_code {
                "authorization_pending" => continue,
                "slow_down" => {
                    interval_secs = interval_secs.saturating_add(5);
                    continue;
                }
                "expired_token" => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: "device code expired before authorization completed".into(),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
                "access_denied" => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: "authorization was denied by user".into(),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
                other => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: format!("device token error HTTP {status}: {other}"),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
            }
        }
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("device token poll HTTP {status}: {text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
}

pub(super) async fn exchange_refresh_token(
    profile: &str,
    current: &OAuthStoredToken,
    oauth_cfg: Option<&OAuthConfig>,
) -> Result<OAuthStoredToken> {
    let refresh_token = current
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: format!("stored token for `{profile}` has no refresh_token"),
            retryable: false,
            retry_after_ms: None,
        })?;
    let token_url = current
        .token_url
        .as_deref()
        .or_else(|| oauth_cfg.and_then(|c| c.token_url.as_deref()))
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: format!("no OAuth `token_url` configured to refresh `{profile}`"),
            retryable: false,
            retry_after_ms: None,
        })?;
    let client_id = current
        .client_id
        .clone()
        .or_else(|| oauth_cfg.and_then(|c| c.resolve_client_id(None)));
    let client_secret = oauth_cfg.and_then(OAuthConfig::resolve_client_secret);

    let mut form: Vec<(String, String)> = vec![
        ("grant_type".into(), "refresh_token".into()),
        ("refresh_token".into(), refresh_token.to_string()),
    ];
    if let Some(ref cid) = client_id {
        form.push(("client_id".into(), cid.clone()));
    }
    if let Some(secret) = client_secret {
        form.push(("client_secret".into(), secret));
    }
    if let Some(cfg) = oauth_cfg {
        for (k, v) in &cfg.extra_token_params {
            form.push((k.clone(), v.clone()));
        }
    }

    let client = build_http_client_for_url(token_url);
    let resp = client
        .post(token_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token refresh failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;

    parse_token_response(
        profile,
        resp,
        client_id.as_deref(),
        Some(token_url),
        Some(refresh_token),
    )
    .await
}

async fn parse_token_response(
    profile: &str,
    resp: reqwest::Response,
    client_id: Option<&str>,
    token_url: Option<&str>,
    fallback_refresh_token: Option<&str>,
) -> Result<OAuthStoredToken> {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token endpoint HTTP {status}: {text}"),
            retryable: status.is_server_error(),
            retry_after_ms: None,
        });
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid JSON from OAuth token endpoint: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;
    parse_token_value(profile, &v, client_id, token_url, fallback_refresh_token)
}

pub(super) fn parse_token_value(
    profile: &str,
    v: &Value,
    client_id: Option<&str>,
    token_url: Option<&str>,
    fallback_refresh_token: Option<&str>,
) -> Result<OAuthStoredToken> {
    let access_token = v
        .get("access_token")
        .or_else(|| v.get("key"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "OAuth token response missing `access_token`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();

    let refresh_token = v
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .or_else(|| fallback_refresh_token.map(ToString::to_string));

    let token_type = v
        .get("token_type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("Bearer")
        .to_string();

    let now = now_epoch_secs();
    let expires_at_epoch_secs = v
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|secs| now.saturating_add(secs));

    let scope = v
        .get("scope")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    Ok(OAuthStoredToken {
        access_token,
        refresh_token,
        token_type,
        expires_at_epoch_secs,
        scope,
        client_id: client_id.map(ToString::to_string),
        token_url: token_url.map(ToString::to_string),
        updated_at_epoch_secs: now,
    })
}
