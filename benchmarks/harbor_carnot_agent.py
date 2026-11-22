"""
Harbor Agent Adapter for Carnot (`carnot-cli`).

Usage with Harbor (Terminal-Bench Hard / Terminal-Bench 2.0 / SWE-bench):

    PYTHONPATH=/path/to/carnot/benchmarks \
      harbor run \
        -a harbor_carnot_agent:CarnotHarborAgent \
        -d cohere/terminal_bench_aai \
        -m anthropic/claude-sonnet-5-5 \
        --ak reasoning_effort=medium \
        -n 4

Supports both `linux/aarch64` and `linux/x86_64` containers via static musl binaries
built in `/tmp/carnot-harbor-bin/` (override with `CARNOT_HARBOR_BIN_DIR` or `--ak bin_dir=...`).
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from harbor.agents.agent_auth_resolution import ModelConnectionSpec, ProviderAccess
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage
from harbor.models.model_api_format import ModelApiFormat

DEFAULT_BIN_DIR = Path(os.environ.get("CARNOT_HARBOR_BIN_DIR", "/tmp/carnot-harbor-bin"))

FORWARD_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "XAI_BASE_URL",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "MUSE_BASE_URL",
    "MUSE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
)


class CarnotHarborOptions(AgentOptions):
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default="medium",
        description="Carnot --effort level (low, medium, high, xhigh, max).",
    )
    cache_ttl: Literal["hybrid", "5m", "1h"] = Field(
        default="hybrid",
        description="Carnot --cache-ttl policy (hybrid, 5m, 1h).",
    )
    bin_dir: str | None = Field(
        default=None,
        description="Directory containing carnot-linux-aarch64 / carnot-linux-x86_64 binaries.",
    )
    profile: Literal["minimal", "standard", "carnot"] = Field(
        default="carnot",
        description="Carnot module profile preset.",
    )
    max_turns: int | None = Field(
        default=None,
        description="Optional override for Carnot --max-turns (default 500).",
    )
    subagent: bool = Field(
        default=False,
        description="Enable Carnot disposable-context Subagent tool.",
    )


class CarnotHarborAgent(BaseAgent):
    capabilities: ClassVar[AgentCapabilities] = AgentCapabilities(
        base_url=True,
        reasoning_effort=True,
        skills=True,
        mcp_servers=True,
    )
    options_model: ClassVar[type[AgentOptions] | None] = CarnotHarborOptions

    MODEL_CONNECTION: ClassVar[ModelConnectionSpec | None] = ModelConnectionSpec(
        default_provider="anthropic",
        agent_provider_access=ProviderAccess(
            api_key_candidate_names=[
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "XAI_API_KEY",
                "GEMINI_API_KEY",
            ],
            base_url_candidate_names=["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"],
        ),
        api_key_destination_names={"default": "ANTHROPIC_API_KEY"},
        base_url_destination_names={"default": "ANTHROPIC_BASE_URL"},
        supported_api_formats=[
            ModelApiFormat.ANTHROPIC_MESSAGES,
            ModelApiFormat.OPENAI_CHAT_COMPLETIONS,
            ModelApiFormat.OPENAI_RESPONSES,
        ],
        default_api_format_by_provider={
            "anthropic": ModelApiFormat.ANTHROPIC_MESSAGES,
            "openai": ModelApiFormat.OPENAI_RESPONSES,
            "default": ModelApiFormat.ANTHROPIC_MESSAGES,
        },
    )

    @staticmethod
    def name() -> str:
        return "carnot"

    def version(self) -> str | None:
        return "0.1.0-v8.0"

    def _carnot_options(self) -> CarnotHarborOptions:
        if isinstance(self.options, CarnotHarborOptions):
            return self.options
        return CarnotHarborOptions()

    def _resolve_provider_and_model(self) -> tuple[str, str]:
        raw = (self.model_name or "anthropic/claude-sonnet-5-5").strip()
        if "/" in raw:
            prov, model = raw.split("/", 1)
            return prov.strip().lower(), model.strip()
        return "anthropic", raw

    async def setup(self, environment: BaseEnvironment) -> None:
        opts = self._carnot_options()
        bin_dir = Path(opts.bin_dir) if opts.bin_dir else DEFAULT_BIN_DIR

        arch_res = await environment.exec("uname -m", user="root")
        arch = (arch_res.stdout or "").strip()
        if arch in ("aarch64", "arm64"):
            local_bin = bin_dir / "carnot-linux-aarch64"
        elif arch in ("x86_64", "amd64"):
            local_bin = bin_dir / "carnot-linux-x86_64"
        else:
            raise RuntimeError(f"Unsupported container architecture for Carnot: {arch!r}")

        if not local_bin.is_file():
            raise FileNotFoundError(
                f"Carnot Linux binary not found at {local_bin}. "
                f"Build static musl binaries into {bin_dir} first."
            )

        await environment.upload_file(local_bin, "/usr/local/bin/carnot")
        chmod_res = await environment.exec(
            "chmod 755 /usr/local/bin/carnot && /usr/local/bin/carnot --version",
            user="root",
        )
        if chmod_res.return_code != 0:
            raise RuntimeError(
                f"Carnot binary verification failed (rc={chmod_res.return_code}): "
                f"{chmod_res.stdout} {chmod_res.stderr}"
            )

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
        for key in FORWARD_ENV_KEYS:
            val = self._get_env(key)
            if val is not None and val != "":
                env[key] = val
        # Respect explicit --base-url override if supplied to Harbor
        if self._base_url:
            provider, _ = self._resolve_provider_and_model()
            if provider == "anthropic":
                env["ANTHROPIC_BASE_URL"] = self._base_url
            else:
                env["OPENAI_BASE_URL"] = self._base_url
        return env

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        opts = self._carnot_options()
        provider, model = self._resolve_provider_and_model()
        effort = opts.reasoning_effort or "medium"

        # Upload prompt via temp file (0644 so non-root container users can read it!)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write(instruction)
            prompt_host_path = Path(tf.name)
        try:
            os.chmod(prompt_host_path, 0o644)
            await environment.upload_file(prompt_host_path, "/tmp/.carnot-prompt.txt")
        finally:
            prompt_host_path.unlink(missing_ok=True)

        extra_flags: list[str] = []
        if opts.max_turns is not None:
            extra_flags.append(f"--max-turns {int(opts.max_turns)}")
        if opts.subagent:
            extra_flags.append("--subagent")
        if self.skills_dir:
            extra_flags.append(f"--skills-dir {shlex.quote(str(self.skills_dir))}")

        if self.mcp_servers:
            mcp_map: dict[str, dict] = {}
            for srv in self.mcp_servers:
                transport = getattr(srv, "transport", None)
                url = getattr(srv, "url", None)
                command = getattr(srv, "command", None)
                args = getattr(srv, "args", None) or []
                headers = getattr(srv, "headers", None) or {}
                if url:
                    entry: dict = {"url": str(url), "lazy": False}
                    if transport:
                        entry["transport"] = str(transport)
                    if headers:
                        entry["headers"] = dict(headers)
                    mcp_map[str(srv.name)] = entry
                elif command:
                    mcp_map[str(srv.name)] = {
                        "command": str(command),
                        "args": [str(a) for a in args],
                        "lazy": False,
                    }
            if mcp_map:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as mf:
                    json.dump({"mcpServers": mcp_map}, mf)
                    mcp_host_path = Path(mf.name)
                try:
                    os.chmod(mcp_host_path, 0o644)
                    await environment.upload_file(mcp_host_path, "/tmp/.carnot-mcp.json")
                finally:
                    mcp_host_path.unlink(missing_ok=True)
                extra_flags.append("--mcp-config /tmp/.carnot-mcp.json")

        flags_str = (" " + " ".join(extra_flags)) if extra_flags else ""

        log_dir = str(self.environment_logs_dir)
        jsonl_path = f"{log_dir}/carnot.jsonl"
        stderr_path = f"{log_dir}/carnot.stderr.log"

        cmd = (
            f"mkdir -p {shlex.quote(log_dir)} && "
            f"/usr/local/bin/carnot "
            f"--no-sandbox --no-session "
            f"--mode json "
            f"--profile {shlex.quote(opts.profile)} "
            f"--provider {shlex.quote(provider)} "
            f"--model {shlex.quote(model)} "
            f"--effort {shlex.quote(effort)} "
            f"--cache-ttl {shlex.quote(opts.cache_ttl)}"
            f"{flags_str} "
            f"--prompt-file /tmp/.carnot-prompt.txt "
            f"> {shlex.quote(jsonl_path)} 2> {shlex.quote(stderr_path)}; "
            f"rc=$?; rm -rf .carnot /tmp/.carnot-prompt.txt /tmp/.carnot-mcp.json; exit $rc"
        )

        res = await environment.exec(cmd, env=self._build_env())
        host_jsonl = self.logs_dir / "carnot.jsonl"
        if not host_jsonl.is_file():
            try:
                host_jsonl.parent.mkdir(parents=True, exist_ok=True)
                await environment.download_file(jsonl_path, host_jsonl)
            except Exception:
                pass
        self.populate_context_post_run(context)
        if context.metadata is None:
            context.metadata = {}
        context.metadata["return_code"] = res.return_code
        context.metadata["carnot_effort"] = effort
        context.metadata["carnot_cache_ttl"] = opts.cache_ttl
        context.metadata["carnot_profile"] = opts.profile

    def populate_context_post_run(self, context: AgentContext) -> None:
        jsonl_file = self.logs_dir / "carnot.jsonl"
        if not jsonl_file.is_file():
            return

        in_t = 0
        cr_t = 0
        cw5_t = 0
        cw1_t = 0
        out_t = 0
        cost_usd = 0.0
        turns = 0
        rewinds = 0
        settled_metrics: dict | None = None
        last_error: str | None = None

        try:
            for line in jsonl_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("event")
                if kind == "usage_recorded":
                    turns += 1
                    u = ev.get("usage") or {}
                    in_t += int(u.get("input_tokens", 0) or 0)
                    cr_t += int(u.get("cache_read_tokens", 0) or 0)
                    cw5_t += int(u.get("cache_write_5m_tokens", 0) or 0)
                    cw1_t += int(u.get("cache_write_1h_tokens", 0) or 0)
                    out_t += int(u.get("output_tokens", 0) or 0)
                    cost_usd += float(u.get("cost_usd", 0.0) or 0.0)
                elif kind == "rewind_applied":
                    rewinds += 1
                elif kind == "run_settled":
                    settled_metrics = ev.get("metrics")
                elif kind == "error":
                    last_error = str(ev.get("message", ""))
        except OSError:
            return

        if settled_metrics and isinstance(settled_metrics.get("total_usage"), dict):
            u = settled_metrics["total_usage"]
            in_t = int(u.get("input_tokens", in_t) or 0)
            cr_t = int(u.get("cache_read_tokens", cr_t) or 0)
            cw5_t = int(u.get("cache_write_5m_tokens", cw5_t) or 0)
            cw1_t = int(u.get("cache_write_1h_tokens", cw1_t) or 0)
            out_t = int(u.get("output_tokens", out_t) or 0)
            cost_usd = float(u.get("cost_usd", cost_usd) or 0.0)
            turns = int(settled_metrics.get("turns", turns) or turns)
            rewinds = int(settled_metrics.get("rewinds_applied", rewinds) or rewinds)

        total_in = in_t + cr_t + cw5_t + cw1_t
        if total_in > 0 or out_t > 0:
            context.n_input_tokens = total_in
            context.n_cache_tokens = cr_t
            context.n_output_tokens = out_t
            context.cost_usd = round(cost_usd, 6)
            _, model = self._resolve_provider_and_model()
            context.model_usage = {
                model: ModelUsage(
                    n_input_tokens=total_in,
                    n_cache_tokens=cr_t,
                    n_output_tokens=out_t,
                    cost_usd=round(cost_usd, 6),
                )
            }

        meta = dict(context.metadata or {})
        meta.update(
            {
                "turns": turns,
                "rewinds_applied": rewinds,
                "cache_read_tokens": cr_t,
                "cache_write_5m_tokens": cw5_t,
                "cache_write_1h_tokens": cw1_t,
                "uncached_input_tokens": in_t,
                "cache_hit_rate": round(cr_t / total_in, 4) if total_in > 0 else 0.0,
            }
        )
        if last_error:
            meta["last_error"] = last_error[:400]
        context.metadata = meta
