/**
 * Zero-dependency TypeScript SDK for Carnot (Bearer-authenticated HTTP/SSE + Last-Event-ID + steer).
 */
export class CarnotHttpClient {
    baseUrl;
    token;
    constructor(baseUrl = "http://127.0.0.1:7744", token) {
        this.baseUrl = baseUrl;
        this.token = token;
    }
    authHeaders(extra = {}) {
        const h = { ...extra };
        if (this.token) {
            h["Authorization"] = `Bearer ${this.token}`;
        }
        return h;
    }
    async health() {
        const res = await fetch(`${this.baseUrl}/health`);
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async metrics() {
        const res = await fetch(`${this.baseUrl}/v1/metrics`, {
            headers: this.authHeaders(),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async setCacheTtl(ttl) {
        const res = await fetch(`${this.baseUrl}/v1/cache-ttl`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify({ ttl }),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async goalStatus() {
        const res = await fetch(`${this.baseUrl}/v1/goal`, {
            headers: this.authHeaders(),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async setGoal(input) {
        const res = await fetch(`${this.baseUrl}/v1/goal`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify(input),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async clearGoal() {
        const res = await fetch(`${this.baseUrl}/v1/goal`, {
            method: "DELETE",
            headers: this.authHeaders(),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async prompt(prompt, images) {
        const body = images?.length ? { prompt, images, stream: false } : { prompt, stream: false };
        const res = await fetch(`${this.baseUrl}/v1/prompt`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify(body),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        return res.json();
    }
    async steer(message) {
        const res = await fetch(`${this.baseUrl}/v1/steer`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify({ message }),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
    }
    async followUp(message) {
        const res = await fetch(`${this.baseUrl}/v1/follow-up`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify({ message }),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
    }
    async compact(focus) {
        const res = await fetch(`${this.baseUrl}/v1/compact`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify(focus ? { focus } : {}),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async undo() {
        const res = await fetch(`${this.baseUrl}/v1/undo`, {
            method: "POST",
            headers: this.authHeaders(),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async *promptStream(prompt, images) {
        const body = images?.length ? { prompt, images, stream: true } : { prompt, stream: true };
        const res = await fetch(`${this.baseUrl}/v1/prompt`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify(body),
        });
        if (!res.ok || !res.body)
            throw new Error(`HTTP ${res.status}`);
        yield* parseSseStream(res.body);
    }
    async *events(lastEventId = 0) {
        const res = await fetch(`${this.baseUrl}/v1/events`, {
            headers: this.authHeaders({ "Last-Event-ID": String(lastEventId) }),
        });
        if (!res.ok || !res.body)
            throw new Error(`HTTP ${res.status}`);
        yield* parseSseStream(res.body);
    }
    async listPending() {
        const res = await fetch(`${this.baseUrl}/v1/pending`, {
            headers: this.authHeaders(),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async respondPermission(requestId, decision, reason) {
        const res = await fetch(`${this.baseUrl}/v1/permissions`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify({ request_id: requestId, decision, reason }),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async respondQuestion(requestId, answers) {
        const res = await fetch(`${this.baseUrl}/v1/questions`, {
            method: "POST",
            headers: this.authHeaders({ "content-type": "application/json" }),
            body: JSON.stringify({ request_id: requestId, answers }),
        });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return res.json();
    }
    async abort() {
        await fetch(`${this.baseUrl}/v1/abort`, {
            method: "POST",
            headers: this.authHeaders(),
        });
    }
}
async function* parseSseStream(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done)
            break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
            const frame = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            const line = frame.split("\n").find((l) => l.startsWith("data:"));
            if (line) {
                yield JSON.parse(line.slice(5).trim());
            }
        }
    }
}
