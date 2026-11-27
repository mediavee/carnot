#!/usr/bin/env python3
"""Generates light & dark Carnot footprint comparison SVGs directly from compare_harnesses.py JSON."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any


THEMES = {
    "dark": {
        "bg": "#0D1017",
        "card": "#131722",
        "border": "rgba(255,255,255,0.09)",
        "track": "rgba(255,255,255,0.06)",
        "title": "#F3F4F6",
        "muted": "#8B95A7",
        "bar_carnot_start": "#FF9A3E",
        "bar_carnot_end": "#FF4D1C",
        "bar_other": "#334155",
        "val_carnot": "#FF8C42",
        "val_other": "#CBD5E1",
        "label_carnot": "#FFFFFF",
        "label_other": "#94A3B8",
        "badge_bg": "rgba(255,77,28,0.14)",
        "badge_text": "#FF8C42",
    },
    "light": {
        "bg": "#F8FAFC",
        "card": "#FFFFFF",
        "border": "rgba(15,23,42,0.10)",
        "track": "rgba(15,23,42,0.06)",
        "title": "#0F172A",
        "muted": "#64748B",
        "bar_carnot_start": "#FF8A2B",
        "bar_carnot_end": "#EA3E14",
        "bar_other": "#CBD5E1",
        "val_carnot": "#D93812",
        "val_other": "#334155",
        "label_carnot": "#0F172A",
        "label_other": "#475569",
        "badge_bg": "rgba(234,62,20,0.10)",
        "badge_text": "#C83210",
    },
}

SHORT_LABELS = {
    "carnot-core": "Carnot (core)",
    "carnot": "Carnot (full)",
    "codex": "Codex CLI",
    "claude": "Claude Code",
    "pi": "Pi (Node)",
    "opencode": "OpenCode",
}


def fmt_val(v: float, unit: str) -> str:
    if unit == "tok":
        return f"{int(round(v)):,}".replace(",", "\u2009") + " tok"
    if v >= 100:
        return f"{v:.0f} {unit}"
    return f"{v:.1f} {unit}"


def render_svg(data: dict[str, Any], mode: str) -> str:
    t = THEMES[mode]
    rows = [r for r in data.get("results", []) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit("No successful harness rows in input JSON")

    panels = [
        ("Cold Start (median)", "startup_first_http_median_ms", "ms"),
        ("Peak Memory (RSS)", "peak_rss_mib", "MiB"),
        ("Installed Footprint", "closure_mb", "MB"),
        ("Static Prefix Overhead", "static_prefix_est_tokens", "tok"),
    ]

    n = len(rows)
    row_h = 30
    panel_pad_top = 46
    panel_pad_bot = 16
    panel_h = panel_pad_top + n * row_h + panel_pad_bot
    panel_w = 392
    gap_x = 16
    gap_y = 16
    outer_pad = 14
    width = outer_pad * 2 + panel_w * 2 + gap_x
    footer_h = 28
    height = outer_pad * 2 + panel_h * 2 + gap_y + footer_h

    label_x = 16
    bar_x = 112
    bar_max_w = 178
    val_x = panel_w - 16

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Carnot embedded harness footprint comparison">',
        "  <defs>",
        f'    <linearGradient id="carnotGrad" x1="0" y1="0" x2="1" y2="0">',
        f'      <stop offset="0%" stop-color="{t["bar_carnot_start"]}"/>',
        f'      <stop offset="100%" stop-color="{t["bar_carnot_end"]}"/>',
        "    </linearGradient>",
        "    <style>",
        '      .font { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif; }',
        '      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        "    </style>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="16" fill="{t["bg"]}"/>',
    ]

    for p_idx, (title, key, unit) in enumerate(panels):
        col = p_idx % 2
        row = p_idx // 2
        px = outer_pad + col * (panel_w + gap_x)
        py = outer_pad + row * (panel_h + gap_y)

        tie_order = {"carnot-core": 0, "carnot": 1, "codex": 2, "pi": 3, "claude": 4, "opencode": 5}
        sorted_rows = sorted(
            rows,
            key=lambda r: (float(r.get(key) or 0.0), tie_order.get(r["id"], 99)),
        )
        max_v = max(float(r.get(key) or 0.0) for r in sorted_rows) or 1.0

        parts.append(f'  <g transform="translate({px},{py})">')
        parts.append(
            f'    <rect width="{panel_w}" height="{panel_h}" rx="12" fill="{t["card"]}" stroke="{t["border"]}" stroke-width="1"/>'
        )
        parts.append(
            f'    <text x="16" y="28" class="font" font-size="13" font-weight="650" fill="{t["title"]}">{html.escape(title)}</text>'
        )
        parts.append(
            f'    <text x="{panel_w - 16}" y="27" text-anchor="end" class="mono" font-size="10.5" fill="{t["muted"]}">lower is better</text>'
        )

        for r_idx, r in enumerate(sorted_rows):
            hid = r["id"]
            is_carnot = hid.startswith("carnot")
            label = SHORT_LABELS.get(hid, r.get("label", hid))
            val = float(r.get(key) or 0.0)
            frac = max(0.025, min(1.0, val / max_v))
            bw = round(frac * bar_max_w, 1)
            ry = panel_pad_top + r_idx * row_h

            lbl_fill = t["label_carnot"] if is_carnot else t["label_other"]
            lbl_weight = "700" if is_carnot else "500"
            bar_fill = "url(#carnotGrad)" if is_carnot else t["bar_other"]
            val_fill = t["val_carnot"] if is_carnot else t["val_other"]
            val_weight = "700" if is_carnot else "500"

            parts.append(
                f'    <text x="{label_x}" y="{ry + 15}" class="font" font-size="12" font-weight="{lbl_weight}" fill="{lbl_fill}">{html.escape(label)}</text>'
            )
            parts.append(
                f'    <rect x="{bar_x}" y="{ry + 5}" width="{bar_max_w}" height="12" rx="4" fill="{t["track"]}"/>'
            )
            parts.append(
                f'    <rect x="{bar_x}" y="{ry + 5}" width="{bw}" height="12" rx="4" fill="{bar_fill}"/>'
            )
            parts.append(
                f'    <text x="{val_x}" y="{ry + 15}" text-anchor="end" class="mono" font-size="11.5" font-weight="{val_weight}" fill="{val_fill}">{html.escape(fmt_val(val, unit))}</text>'
            )

        parts.append("  </g>")

    plat = data.get("platform", "")
    runs = data.get("runs_per_harness", 15)
    warmups = data.get("warmups_per_harness", 3)
    footer_y = height - 12
    footer_txt = (
        f"Measured locally via benchmarks/compare_harnesses.py · {runs} runs + {warmups} warmups · {plat} · $0 mock API"
    )
    parts.append(
        f'  <text x="{width / 2:.1f}" y="{footer_y}" text-anchor="middle" class="mono" font-size="10.5" fill="{t["muted"]}">{html.escape(footer_txt)}</text>'
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: render_footprint_svg.py <results.json> [assets_dir]", file=sys.stderr)
        return 2
    in_path = Path(sys.argv[1])
    assets_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "assets"
    data = json.loads(in_path.read_text(encoding="utf-8"))
    for mode in ("dark", "light"):
        out = assets_dir / f"benchmarks-{mode}.svg"
        out.write_text(render_svg(data, mode), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
