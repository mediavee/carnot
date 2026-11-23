/**
 * Zero-dependency TypeScript SDK for Carnot (Bearer-authenticated HTTP/SSE + Last-Event-ID + steer).
 */

export interface ThermodynamicMetrics {
  turns: number;
  requests: number;
  context_auc_tokens: number;
  compactions_applied: number;
  heat_shed_tokens: number;
  cas_handles_created: number;
  prefix_cache_hits: number;
  prefix_cache_misses: number;
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

export interface EventEnvelope {
  seq: number;
  event: string;
  [key: string]: unknown;
}

export interface PromptResponse {
  final_text: string;
  metrics: ThermodynamicMetrics;
  halted_reason?: string | null;
}

export class CarnotHttpClient {
  constructor(
    private readonly baseUrl = "http://127.0.0.1:7744",
    private readonly token?: string,
  ) {}

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
    metrics: ThermodynamicMetrics;
  }> {
    const res = await fetch(`${this.baseUrl}/v1/metrics`, {
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async prompt(prompt: string): Promise<PromptResponse> {
    const res = await fetch(`${this.baseUrl}/v1/prompt`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ prompt, stream: false }),
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

  async *promptStream(prompt: string): AsyncGenerator<EventEnvelope> {
    const res = await fetch(`${this.baseUrl}/v1/prompt`, {
      method: "POST",
      headers: this.authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ prompt, stream: true }),
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
