import { useState } from "react";
import type { AgentTraceEntry } from "../lib/types";

export const AGENT_LABELS: Record<string, string> = {
  asr: "🎙️ Speech-to-text",
  vision: "📄 Document OCR",
  combine: "🔗 Combine inputs",
  planner: "🧭 Planner (topic routing)",
  retrieval: "📚 Retrieval (statute search)",
  reasoning: "🧠 Reasoning (drafts the answer)",
  grounding: "✅ Grounding (fact-checks against law)",
  risk: "⚠️ Risk (deadline + confidence)",
  escalation: "🤝 Escalation (free-aid eligibility)",
  output: "📦 Output (assembles the answer)",
  translation: "🌐 Translation",
  tts: "🔊 Text-to-speech",
};

function labelFor(agent: string): string {
  return AGENT_LABELS[agent] ?? agent;
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v || "—";
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

function AgentCard({ entry }: { entry: AgentTraceEntry }) {
  const [open, setOpen] = useState(false);
  const outputEntries = Object.entries(entry.output ?? {});

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 p-2.5 text-left text-sm"
      >
        <span className="flex items-center gap-2">
          <span className={entry.status === "ok" ? "text-green-600" : "text-red-600"}>
            {entry.status === "ok" ? "✅" : "⚠️"}
          </span>
          <span className="font-medium text-slate-700">{labelFor(entry.agent)}</span>
        </span>
        <span className="flex items-center gap-2 text-xs text-slate-400">
          {entry.duration_ms.toFixed(0)}ms
          <span>{open ? "▲" : "▼"}</span>
        </span>
      </button>
      {open && (
        <div className="border-t border-slate-100 p-2.5 text-xs text-slate-600">
          {outputEntries.length === 0 ? (
            <div className="text-slate-400">No output (no-op for this request)</div>
          ) : (
            <dl className="flex flex-col gap-1">
              {outputEntries.map(([k, v]) => (
                <div key={k} className="flex flex-col gap-0.5 sm:flex-row sm:gap-2">
                  <dt className="shrink-0 font-mono font-semibold text-slate-500">{k}:</dt>
                  <dd className="whitespace-pre-wrap break-words font-mono">{formatValue(v)}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      )}
    </div>
  );
}

interface Props {
  agentTrace: AgentTraceEntry[];
  signalPacket?: Record<string, unknown>;
}

export default function AgentTrace({ agentTrace, signalPacket }: Props) {
  if (agentTrace.length === 0) return null;

  const okCount = agentTrace.filter((e) => e.status === "ok").length;
  const totalMs = agentTrace.reduce((sum, e) => sum + e.duration_ms, 0);

  return (
    <section className="flex flex-col gap-2">
      <p className="text-xs text-slate-400">
        {okCount}/{agentTrace.length} steps completed · {(totalMs / 1000).toFixed(1)}s total
      </p>
      <div className="flex flex-col gap-1.5">
        {agentTrace.map((entry, i) => (
          <AgentCard key={`${entry.agent}-${i}`} entry={entry} />
        ))}
      </div>
      {signalPacket && Object.keys(signalPacket).length > 0 && (
        <details className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs">
          <summary className="cursor-pointer font-medium text-slate-600">Raw signal packet (UNO Q)</summary>
          <pre className="mt-2 overflow-x-auto rounded bg-slate-100 p-2">
            {JSON.stringify(signalPacket, null, 2)}
          </pre>
        </details>
      )}
    </section>
  );
}
