import type { AgentTraceEntry, AskRequest, AskResponse, HealthResponse, KbStoresResponse } from "./types";

// Dev server proxies /api -> the FastAPI backend (see vite.config.ts). In
// production, point VITE_API_BASE_URL at wherever api.py is deployed.
const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`JusticeBridge API ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getKbStores(): Promise<KbStoresResponse> {
  return request<KbStoresResponse>("/kb-stores");
}

export function ask(payload: AskRequest): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// On-device inference has no result until the whole graph finishes (20s-2min+),
// which reads as "frozen" over one blocking request. /ask/stream sends one
// newline-delimited JSON line per agent as it completes, so the UI can show
// live progress instead of a blank wait. fetch()'s ReadableStream (not
// EventSource, which is GET-only and can't carry a POST body) reads the
// response as it arrives; each complete line is parsed and handed to onStep
// as soon as the network delivers it — no buffering for the full response.
export async function askStream(
  payload: AskRequest,
  onStep: (step: AgentTraceEntry) => void
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    throw new Error(`JusticeBridge API /ask/stream failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: AskResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIdx: number;
    while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (!line) continue;
      const msg = JSON.parse(line) as
        | { type: "agent_step"; step: AgentTraceEntry }
        | { type: "done"; result: AskResponse };
      if (msg.type === "agent_step") {
        onStep(msg.step);
      } else if (msg.type === "done") {
        result = msg.result;
      }
    }
  }

  if (!result) {
    throw new Error("Stream ended without a final result");
  }
  return result;
}

// Strip the "data:<mime>;base64," prefix FileReader adds — the API wants raw base64.
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      resolve(result.split(",", 2)[1] ?? "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
