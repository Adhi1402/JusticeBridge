import { useEffect, useState } from "react";
import { getHealth } from "../lib/api";
import type { HealthResponse } from "../lib/types";

/** Thin status strip — tells the user (or a developer) whether the AI
 * backend is actually reachable before they try to submit anything. */
export default function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setUnreachable(true));
  }, []);

  if (unreachable) {
    return (
      <div className="bg-red-600 px-4 py-1.5 text-center text-xs font-medium text-white">
        Can't connect right now — please try again in a moment
      </div>
    );
  }
  if (!health) return null;

  const details = [
    health.llm_model ?? health.llm_backend,
    `${health.asr_backend} ASR`,
    `${health.vision_backend} OCR`,
    health.translation_backend && `${health.translation_backend} translation`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div
      className="flex items-center justify-center gap-1.5 bg-slate-100 px-4 py-1.5 text-center text-xs text-slate-500"
      title={details}
    >
      <span className={health.llm_live ? "text-green-600" : "text-amber-600"}>
        {health.llm_live ? "●" : "○"}
      </span>
      {health.llm_live ? "Ready" : "Starting up…"}
      {health.offline && " · 🔒 Works fully offline"}
    </div>
  );
}
