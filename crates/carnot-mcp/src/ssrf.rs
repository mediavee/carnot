//! SSRF & DNS-rebinding defense shared by Streamable HTTP MCP and `WebFetch`.
//!
//! Validates every resolved IP inside a custom `reqwest::dns::Resolve` implementation so the
//! exact socket address connected is guaranteed to be public (zero TOCTOU DNS rebinding!).

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use url::{Host, Url};

/// Returns true if `ip` is loopback, private, link-local, CGNAT, multicast, broadcast,
/// unspecified, or an IPv6 tunneling wrapper (`::ffff:`, `64:ff9b::/96`, `2002::/16`) around one.
pub fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

fn is_blocked_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_unspecified()
        || o[0] == 0 // 0.0.0.0/8
        || (o[0] == 100 && (o[1] & 0b1100_0000) == 64) // 100.64.0.0/10 CGNAT
        || (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2)) // IETF protocol / TEST-NET-1
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19)) // benchmarking 198.18.0.0/15
        || (o[0] == 198 && o[1] == 51 && o[2] == 100) // TEST-NET-2
        || (o[0] == 203 && o[1] == 0 && o[2] == 113) // TEST-NET-3
        || o[0] >= 240 // reserved / class E
}

fn is_blocked_v6(ip: Ipv6Addr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return true;
    }
    let segs = ip.segments();
    // Unique local fc00::/7
    if (segs[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    // Link-local fe80::/10
    if (segs[0] & 0xffc0) == 0xfe80 {
        return true;
    }
    // IPv4-mapped ::ffff:a.b.c.d
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_blocked_v4(v4);
    }
    // NAT64 well-known prefix 64:ff9b::/96 (and 64:ff9b:1::/48)
    if segs[0] == 0x0064 && segs[1] == 0xff9b {
        let v4 = Ipv4Addr::new(
            (segs[6] >> 8) as u8,
            (segs[6] & 0xff) as u8,
            (segs[7] >> 8) as u8,
            (segs[7] & 0xff) as u8,
        );
        return is_blocked_v4(v4);
    }
    // 6to4 prefix 2002::/16 embeds IPv4 in segs[1..=2]
    if segs[0] == 0x2002 {
        let v4 = Ipv4Addr::new(
            (segs[1] >> 8) as u8,
            (segs[1] & 0xff) as u8,
            (segs[2] >> 8) as u8,
            (segs[2] & 0xff) as u8,
        );
        return is_blocked_v4(v4);
    }
    false
}

/// Validates URL scheme (`http`/`https`) and rejects literal private/loopback hosts up front.
pub fn validate_public_url(raw: &str) -> std::result::Result<Url, String> {
    let parsed = Url::parse(raw).map_err(|e| format!("invalid URL `{raw}`: {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "disallowed URL scheme `{}` (only http/https allowed)",
            parsed.scheme()
        ));
    }
    match parsed.host() {
        Some(Host::Ipv4(v4)) if is_blocked_v4(v4) => {
            Err(format!("SSRF blocked literal IPv4 `{v4}`"))
        }
        Some(Host::Ipv6(v6)) if is_blocked_v6(v6) => {
            Err(format!("SSRF blocked literal IPv6 `{v6}`"))
        }
        Some(Host::Domain(d)) => {
            let low = d.to_ascii_lowercase();
            if low == "localhost" || low.ends_with(".localhost") || low.ends_with(".local") {
                Err(format!("SSRF blocked local domain `{d}`"))
            } else {
                Ok(parsed)
            }
        }
        Some(_) => Ok(parsed),
        None => Err("URL missing host".into()),
    }
}

/// Custom `reqwest` DNS resolver that filters out every private/loopback/link-local IP at connect time.
#[derive(Debug, Default)]
pub struct SsrfSafeDnsResolver;

impl Resolve for SsrfSafeDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?
                .collect();
            let mut safe: Vec<SocketAddr> = Vec::with_capacity(addrs.len());
            for addr in addrs {
                if is_blocked_ip(addr.ip()) {
                    let err: Box<dyn std::error::Error + Send + Sync> =
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            format!("SSRF blocked resolved IP {} for host `{host}`", addr.ip()),
                        ));
                    return Err(err);
                }
                safe.push(addr);
            }
            if safe.is_empty() {
                let err: Box<dyn std::error::Error + Send + Sync> = Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("DNS returned no addresses for `{host}`"),
                ));
                return Err(err);
            }
            let iter: Addrs = Box::new(safe.into_iter());
            Ok(iter)
        })
    }
}

/// Builds a `reqwest::Client` hardened against SSRF, DNS rebinding, and proxy bypasses.
pub fn build_ssrf_safe_client(timeout_secs: u64) -> std::result::Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(SsrfSafeDnsResolver))
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("failed to build SSRF-safe HTTP client: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_private_loopback_cgnat_nat64_and_decimal_ipv4() {
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))));
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1))));
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(is_blocked_ip(IpAddr::V6(
            "::ffff:127.0.0.1".parse().expect("ip")
        )));
        assert!(is_blocked_ip(IpAddr::V6(
            "64:ff9b::169.254.169.254".parse().expect("ip")
        )));
        assert!(!is_blocked_ip(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
        // Decimal IPv4 normalization via url::Host (`2130706433` == 127.0.0.1!)
        assert!(validate_public_url("http://2130706433/admin").is_err());
        assert!(validate_public_url("file:///etc/passwd").is_err());
        assert!(validate_public_url("https://example.com/docs").is_ok());
    }
}
