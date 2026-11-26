/**
 * Zero-dependency TypeScript SDK for Carnot (Bearer-authenticated HTTP/SSE + Last-Event-ID + steer).
 */
export type CacheRetention = "short" | "hybrid" | "long";
export type GoalStatus = "idle" | "running" | "met" | "unreachable" | "stagnated" | "max_iterations" | "cleared";
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
export declare class CarnotHttpClient {
    private readonly baseUrl;
    private readonly token;
    constructor(baseUrl?: string, token?: string);
    private authHeaders;
    health(): Promise<{
        status: string;
        engine: string;
        version: string;
    }>;
    metrics(): Promise<{
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
    }>;
    setCacheTtl(ttl: CacheRetention): Promise<{
        cache_ttl: CacheRetention;
    }>;
    goalStatus(): Promise<GoalState>;
    setGoal(input: GoalConfigInput): Promise<GoalState>;
    clearGoal(): Promise<GoalState>;
    prompt(prompt: string, images?: ImageAttachment[]): Promise<PromptResponse>;
    steer(message: string): Promise<void>;
    followUp(message: string): Promise<void>;
    compact(focus?: string): Promise<{
        tokens_before: number;
        tokens_after: number;
        semantic_llm: boolean;
    }>;
    undo(): Promise<{
        reverted: string | null;
    }>;
    promptStream(prompt: string, images?: ImageAttachment[]): AsyncGenerator<EventEnvelope>;
    events(lastEventId?: number): AsyncGenerator<EventEnvelope>;
    listPending(): Promise<{
        permissions: unknown[];
        questions: unknown[];
    }>;
    respondPermission(requestId: string, decision: "allow_once" | "allow_always" | "deny", reason?: string): Promise<{
        resolved: boolean;
    }>;
    respondQuestion(requestId: string, answers: Array<{
        id: string;
        selected?: string[];
        text?: string;
    }>): Promise<{
        resolved: boolean;
    }>;
    abort(): Promise<void>;
}
