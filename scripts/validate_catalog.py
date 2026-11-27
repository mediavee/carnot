#!/usr/bin/env python3
"""Zero-dependency schema & integrity validator for crates/carnot-ai/catalog/providers.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

KNOWN_APIS = {
    "anthropic-messages",
    "openai-completions",
    "openai-responses",
    "google-generative",
    "google-cloudcode",
    "subprocess-jsonl",
    "faux",
}


def validate_catalog(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"failed to parse {path}: {exc}"]

    if not isinstance(data, dict):
        return ["catalog root must be a JSON object"]

    if data.get("schema_version") != 1:
        errors.append(f"expected schema_version == 1, got {data.get('schema_version')!r}")

    providers = data.get("providers")
    if not isinstance(providers, dict) or not providers:
        errors.append("`providers` must be a non-empty object")
        return errors

    seen_names: set[str] = set()
    for pid, entry in providers.items():
        if not isinstance(entry, dict):
            errors.append(f"provider `{pid}` must be an object")
            continue
        api = entry.get("api")
        if api not in KNOWN_APIS:
            errors.append(f"provider `{pid}` has unknown api `{api}`")
        desc = entry.get("description", "")
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"provider `{pid}` must have a non-empty description")

        pid_low = pid.lower()
        if pid_low in seen_names:
            errors.append(f"duplicate provider id `{pid}`")
        seen_names.add(pid_low)

        for alias in entry.get("aliases", []):
            if not isinstance(alias, str) or not alias.strip():
                errors.append(f"provider `{pid}` has empty alias")
                continue
            al = alias.lower()
            if al in seen_names:
                errors.append(f"provider `{pid}` alias `{alias}` collides with an existing provider/alias")
            seen_names.add(al)

        if api not in {"faux", "subprocess-jsonl"}:
            base_url = entry.get("base_url", "")
            if not isinstance(base_url, str) or not (
                base_url.startswith("https://") or base_url.startswith("http://")
            ):
                errors.append(f"provider `{pid}` must have an http(s) base_url, got `{base_url}`")

        for idx, route in enumerate(entry.get("model_routes", [])):
            if not isinstance(route, dict):
                errors.append(f"provider `{pid}` model_routes[{idx}] must be an object")
                continue
            prefixes = route.get("prefixes")
            if not isinstance(prefixes, list) or not prefixes:
                errors.append(f"provider `{pid}` model_routes[{idx}] must have non-empty `prefixes`")
            r_api = route.get("api")
            if r_api is not None and r_api not in KNOWN_APIS:
                errors.append(f"provider `{pid}` model_routes[{idx}] has unknown api `{r_api}`")

    models = data.get("models", {})
    if not isinstance(models, dict):
        errors.append("`models` must be an object")
    else:
        for mid, m in models.items():
            if not isinstance(m, dict):
                errors.append(f"model `{mid}` must be an object")
                continue
            m_prov = m.get("provider")
            if m_prov is not None and m_prov not in providers:
                errors.append(f"model `{mid}` references unknown provider `{m_prov}`")
            cw = m.get("context_window")
            if cw is not None and (not isinstance(cw, int) or cw <= 0):
                errors.append(f"model `{mid}` context_window must be a positive int, got `{cw}`")
            pricing = m.get("pricing")
            if pricing is not None:
                if not isinstance(pricing, dict):
                    errors.append(f"model `{mid}` pricing must be an object")
                else:
                    for field in (
                        "input_per_mtok",
                        "output_per_mtok",
                        "cache_read_mult",
                        "cache_write_5m_mult",
                        "cache_write_1h_mult",
                    ):
                        val = pricing.get(field)
                        if val is not None and (
                            not isinstance(val, (int, float))
                            or not math.isfinite(val)
                            or val < 0
                        ):
                            errors.append(f"model `{mid}` pricing.{field} invalid: `{val}`")

    return errors


def main() -> int:
    catalog_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent
        / "crates"
        / "carnot-ai"
        / "catalog"
        / "providers.json"
    )
    errors = validate_catalog(catalog_path)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    print(
        f"OK: {catalog_path.name} valid ({len(data['providers'])} providers; "
        f"{len(data.get('models', {}))} models)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
