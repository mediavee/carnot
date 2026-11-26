//! `carnot auth` subcommand handlers (OAuth login, manual token, status, logout).

use crate::args::{AuthCommands, Cli};
use serde_json::json;
use std::time::Duration;

pub(crate) async fn run_auth_cmd(cli: &Cli, action: &AuthCommands) -> i32 {
    let catalog = match carnot::ProviderCatalog::load_layered(&cli.provider_configs) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("auth catalog error: {e}");
            return 1;
        }
    };
    let Some(store) = carnot::AuthStore::open_default() else {
        eprintln!("auth error: could not determine HOME directory for ~/.carnot/auth.json");
        return 1;
    };

    match action {
        AuthCommands::Status => {
            let stored = store.list().unwrap_or_default();
            let summaries = catalog.summaries();
            let mut rows: Vec<serde_json::Value> = Vec::with_capacity(summaries.len());
            for s in summaries {
                let entry_opt = catalog.find_provider(&s.id).map(|(_, e)| e);
                let entry_oauth = entry_opt.and_then(|e| e.oauth.as_ref());
                let profile = entry_opt.map_or(s.id.as_str(), |e| e.oauth_profile(&s.id));
                let tok = stored.get(profile);
                rows.push(json!({
                    "provider": s.id,
                    "oauth_profile": profile,
                    "api": s.api,
                    "configured": s.configured,
                    "auth_source": s.auth_source,
                    "oauth_flow": entry_oauth.map(|o| format!("{:?}", o.flow).to_lowercase()),
                    "oauth_stored": tok.is_some(),
                    "oauth_expired": tok.map(carnot::OAuthStoredToken::is_expired),
                    "oauth_expires_in_secs": tok.and_then(carnot::OAuthStoredToken::seconds_until_expiry),
                    "oauth_refreshable": tok.map(|t| t.can_refresh(entry_oauth)),
                }));
            }
            let out = json!({
                "store_path": store.path(),
                "providers": rows,
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            0
        }
        AuthCommands::Login {
            provider,
            device,
            manual,
            no_browser,
            port,
            client_id,
            token_stdin,
        } => {
            let (canonical_id, entry) = if let Some((id, e)) = catalog.find_provider(provider) {
                (id.to_string(), e.clone())
            } else {
                eprintln!("auth error: unknown provider `{provider}` (run `carnot providers`)");
                return 1;
            };
            let profile = entry.oauth_profile(&canonical_id).to_string();

            if *token_stdin {
                let mut buf = String::new();
                if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf) {
                    eprintln!("auth error reading token from stdin: {e}");
                    return 1;
                }
                match carnot::save_manual_token(&profile, &buf, &store) {
                    Ok(_) => {
                        println!(
                            "Stored Bearer token for `{canonical_id}` (profile `{profile}`) in {} (mode 0600).",
                            store.path().display()
                        );
                        return 0;
                    }
                    Err(e) => {
                        eprintln!("auth error: {e}");
                        return 1;
                    }
                }
            }

            let Some(oauth_cfg) = entry.oauth.as_ref() else {
                eprintln!(
                    "Provider `{canonical_id}` has no OAuth 2.0 flow configured in providers.json.\n\
                     Either add an `oauth` block in ~/.carnot/providers.json (standard PKCE / Device Code),\n\
                     or store a Bearer token securely with: printf '%s' \"$TOKEN\" | carnot auth login {canonical_id} --token-stdin"
                );
                return 1;
            };

            let opts = carnot::OAuthLoginOptions {
                force_device_flow: *device,
                manual_callback: *manual,
                open_browser: !*no_browser,
                port_override: *port,
                client_id_override: client_id.clone(),
                timeout: Duration::from_secs(300),
            };

            let res = carnot::login_with_oauth(
                &profile,
                oauth_cfg,
                &store,
                &opts,
                |prompt| match prompt {
                    carnot::OAuthPrompt::OpenUrl {
                        url,
                        callback_url,
                        manual_paste,
                    } => {
                        eprintln!("Authenticate `{canonical_id}` by opening:\n  {url}");
                        if manual_paste {
                            if callback_url.is_empty() {
                                eprintln!(
                                    "After approving, paste the displayed authorization code below and press Enter:"
                                );
                            } else {
                                eprintln!(
                                    "After approving, paste the full callback URL (starting with {callback_url}) or code below and press Enter:"
                                );
                            }
                        } else {
                            eprintln!("Waiting for OAuth callback on {callback_url} ...");
                        }
                    }
                    carnot::OAuthPrompt::DeviceCode {
                        verification_uri,
                        verification_uri_complete,
                        user_code,
                    } => {
                        eprintln!("Authenticate `{canonical_id}` via Device Code:");
                        eprintln!("  URL:  {verification_uri}");
                        eprintln!("  Code: {user_code}");
                        if let Some(full) = verification_uri_complete {
                            eprintln!("  Direct link: {full}");
                        }
                        eprintln!("Waiting for approval ...");
                    }
                },
                || {
                    let mut line = String::new();
                    std::io::stdin()
                        .read_line(&mut line)
                        .map_err(carnot::CarnotError::Io)?;
                    Ok(line)
                },
            )
            .await;

            match res {
                Ok(tok) => {
                    let expiry = tok
                        .seconds_until_expiry()
                        .map(|s| format!(" (expires in {s}s)"))
                        .unwrap_or_default();
                    let refreshable = if tok.refresh_token.is_some() {
                        ", auto-refresh enabled"
                    } else {
                        ""
                    };
                    println!(
                        "Authenticated `{canonical_id}` (profile `{profile}`) and saved to {}{expiry}{refreshable}.",
                        store.path().display()
                    );
                    0
                }
                Err(e) => {
                    eprintln!("auth login failed: {e}");
                    1
                }
            }
        }
        AuthCommands::Refresh { provider } => {
            let (canonical_id, entry) = if let Some((id, e)) = catalog.find_provider(provider) {
                (id.to_string(), e.clone())
            } else {
                eprintln!("auth error: unknown provider `{provider}`");
                return 1;
            };
            let profile = entry.oauth_profile(&canonical_id).to_string();
            match carnot::refresh_stored_token(&profile, entry.oauth.as_ref(), &store).await {
                Ok(tok) => {
                    let expiry = tok
                        .seconds_until_expiry()
                        .map(|s| format!(" (valid for {s}s)"))
                        .unwrap_or_default();
                    println!(
                        "Refreshed OAuth token for `{canonical_id}` (profile `{profile}`){expiry}."
                    );
                    0
                }
                Err(e) => {
                    eprintln!("auth refresh failed: {e}");
                    1
                }
            }
        }
        AuthCommands::Logout { provider } => {
            let profile = catalog.find_provider(provider).map_or_else(
                || provider.clone(),
                |(id, e)| e.oauth_profile(id).to_string(),
            );
            match store.remove(&profile) {
                Ok(true) => {
                    println!(
                        "Removed stored credentials for `{profile}` from {}.",
                        store.path().display()
                    );
                    0
                }
                Ok(false) => {
                    println!("No stored credentials found for `{profile}`.");
                    0
                }
                Err(e) => {
                    eprintln!("auth logout failed: {e}");
                    1
                }
            }
        }
    }
}
