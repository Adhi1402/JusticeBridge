import type { AgentTraceEntry } from "../lib/types";
import Logo from "./Logo";

// Internal pipeline stage names (asr/planner/grounding/...) are implementation
// detail, not something a citizen looking for legal help needs to see or
// understand. This groups them into a handful of plain-language phases and
// shows one simple progress bar — the technical per-agent breakdown still
// exists (ResultScreen's "Pipeline internals" panel) for anyone curious, but
// it's opt-in, not the default experience.
const PHASES = [
  { label: "Understanding your question", agents: ["asr", "vision", "combine"] },
  { label: "Finding the relevant law", agents: ["planner", "retrieval"] },
  { label: "Preparing your answer", agents: ["reasoning", "grounding", "risk", "escalation", "output"] },
  { label: "Finishing up", agents: ["translation", "tts"] },
];

const TOTAL_AGENTS = PHASES.flatMap((p) => p.agents).length;

interface Props {
  steps: AgentTraceEntry[];
}

export default function LiveProgress({ steps }: Props) {
  const doneAgents = new Set(steps.map((s) => s.agent));
  const doneCount = Math.min(doneAgents.size, TOTAL_AGENTS);
  const percent = Math.max(6, Math.round((doneCount / TOTAL_AGENTS) * 100));

  const currentPhase =
    PHASES.find((p) => p.agents.some((a) => !doneAgents.has(a))) ?? PHASES[PHASES.length - 1];

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 px-6 py-16 text-center">
      <Logo />
      <div className="h-14 w-14 animate-spin rounded-full border-4 border-slate-200 border-t-[#1e3a5f]" />
      <div>
        <div className="text-base font-semibold text-slate-700">{currentPhase.label}…</div>
        <div className="mt-1 text-sm text-slate-400">This runs on your device — no internet used</div>
      </div>
      <div className="w-full max-w-xs">
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
          <div
            className="h-full rounded-full bg-[#1e3a5f] transition-all duration-500 ease-out"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
      <p className="max-w-xs text-xs text-slate-400">
        Please keep this page open while we prepare your answer.
      </p>
    </div>
  );
}
