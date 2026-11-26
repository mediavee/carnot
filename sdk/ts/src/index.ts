/**
 * Zero-dependency TypeScript SDK for Carnot (Bearer-authenticated HTTP/SSE + Last-Event-ID + steer).
 */

export type CacheRetention = "short" | "hybrid" | "long";

export type GoalStatus =
  | "idle"
  | "running"
  | "met"
  | "unreachable"
  | "stagnated"
  | "max_iterations"
  | "cleared";

export interface GoalState {
  status: GoalStatus;
  criterion: string | null;
  verify_commands: string[];
  iterations: number;
  max_iterations: number;
  last_feedback: string | null;
}

export interface GoalConfigInput {
  criterion?: string;
  verify_commands?: string[];
  max_iterations?: number;
  verifier_model?: string;
}

export interface SessionMetrics {
  turns: number;
  requests: number;
  context_auc_tokens: number;
  compactions_applied: number;
  compacted_tokens_saved: number;
  cas_handles_created: number;
  prefix_cache_hits: number;
  prefix_cache_misses: number;
  cache_breaks?: number;
  cacheable_prefix_tokens?: number;
  retained_prefix_tokens?: number;
  eager_tools_started: number;
  parallel_edits_executed: number;
  total_usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_tokens: number;
    cache_write_5m_tokens: number;
    cache_write_1h_tokens: number;
    cost_usd: number;
  };
}

export interface ContextBreakdown {
  system: number;
  tools: number;
  memory: number;
  messages: number;
  tool_results: number;
  estimated: boolean;
}

export interface ImageAttachment {
  media_type: string;
  data_base64: string;
}

export interface EventEnvelope {
  seq: number;
  event: string;
  [key: string]: unknown;
}

export interface PromptResponse {
  final_text: string;
  metrics: SessionMetrics;
  halted_reason?: string | null;
}

export class CarnotHttpClient {
  private readonly baseUrl: string;
  private readonly token: string | undefined;

  constructor(baseUrl = "http://127.0.0.1:7744", token?: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  private authHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const h: Record<string, string> = { ...extra };
    if (this.token) {
      h["Authorization"] = `Bearer ${this.token}`;
    }
    return h;
  }

  async health(): Promise<{ status: string; engine: string; version: string }> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async metrics(): Promise<{
    session_id: string;
    prefix_hash: string;
    latest_seq: number;
    cache_ttl: CacheRetention;
    cache_hit_rate: number;
    prefix_retention_rate: number;
    context_tokens: number;
    context_window: number;
    context_breakdown: ContextBreakdown | null;
    metrics: SessionMetrics;
  }> {
    const res = await fetch(`${this.baseUrl}/v1/metrics`, {
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async setCacheTtl(ttl: CacheRetention): Promise<{ cache_ttl: CacheRetention }> {
    const res = await fetch(`${this.baseUrl}/v1/cache-ttl`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ ttl }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async goalStatus(): Promise<GoalState> {
    const res = await fetch(`${this.baseUrl}/v1/goal`, {
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async setGoal(input: GoalConfigInput): Promise<GoalState> {
    const res = await fetch(`${this.baseUrl}/v1/goal`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify(input),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async clearGoal(): Promise<GoalState> {
    const res = await fetch(`${this.baseUrl}/v1/goal`, {
      method: "DELETE",
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async prompt(prompt: string, images?: ImageAttachment[]): Promise<PromptResponse> {
    const body = images?.length ? { prompt, images, stream: false } : { prompt, stream: false };
    const res = await fetch(`${this.baseUrl}/v1/prompt`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    return res.json();
  }

  async steer(message: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/v1/steer`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ message }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  }

  async followUp(message: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/v1/follow-up`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ message }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  }

  async compact(focus?: string): Promise<{
    tokens_before: number;
    tokens_after: number;
    semantic_llm: boolean;
  }> {
    const res = await fetch(`${this.baseUrl}/v1/compact`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify(focus ? { focus } : {}),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async undo(): Promise<{ reverted: string | null }> {
    const res = await fetch(`${this.baseUrl}/v1/undo`, {
      method: "POST",
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async *promptStream(prompt: string, images?: ImageAttachment[]): AsyncGenerator<EventEnvelope> {
    const body = images?.length ? { prompt, images, stream: true } : { prompt, stream: true };
    const res = await fetch(`${this.baseUrl}/v1/prompt`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    yield* parseSseStream(res.body);
  }

  async *events(lastEventId = 0): AsyncGenerator<EventEnvelope> {
    const res = await fetch(`${this.baseUrl}/v1/events`, {
      headers: this.authHeaders({ "Last-Event-ID": String(lastEventId) }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    yield* parseSseStream(res.body);
  }

  async listPending(): Promise<{ permissions: unknown[]; questions: unknown[] }> {
    const res = await fetch(`${this.baseUrl}/v1/pending`, {
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async respondPermission(
    requestId: string,
    decision: "allow_once" | "allow_always" | "deny",
    reason?: string,
  ): Promise<{ resolved: boolean }> {
    const res = await fetch(`${this.baseUrl}/v1/permissions`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ request_id: requestId, decision, reason }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async respondQuestion(
    requestId: string,
    answers: Array<{ id: string; selected?: string[]; text?: string }>,
  ): Promise<{ resolved: boolean }> {
    const res = await fetch(`${this.baseUrl}/v1/questions`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ request_id: requestId, answers }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async abort(): Promise<void> {
    await fetch(`${this.baseUrl}/v1/abort`, {
      method: "POST",
      headers: this.authHeaders(),
    });
  }
}

async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<EventEnvelope> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (line) {
        yield JSON.parse(line.slice(5).trim()) as EventEnvelope;
      }
    }
  }
}
