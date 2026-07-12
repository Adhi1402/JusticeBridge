import { useState } from "react";
import type { AskResponse } from "../lib/types";
import SeverityBanner from "./SeverityBanner";
import AgentTrace from "./AgentTrace";
import Logo from "./Logo";

interface Props {
  result: AskResponse;
  onStartOver: () => void;
}

function Collapsible({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between p-3 text-left text-sm font-medium text-slate-700"
      >
        {title}
        <span className="text-slate-400">{open ? "▲" : "▼"}</span>
      </button>
      {open && <div className="border-t border-slate-100 p-3 text-sm text-slate-600">{children}</div>}
    </div>
  );
}

export default function ResultScreen({ result, onStartOver }: Props) {
  const answer = result.final_answer_local || result.final_answer_en || "";
  const reasons = result.eligibility_reasons ?? [];
  const dlsa = result.dlsa_contact;
  const audioSrc = result.audio_response_base64
    ? `data:audio/wav;base64,${result.audio_response_base64}`
    : null;

  return (
    <div className="flex flex-col gap-5 px-4 pb-10 pt-4">
      <div className="flex items-center justify-between">
        <Logo size="sm" />
        <button type="button" onClick={onStartOver} className="text-sm text-[#1e3a5f] underline">
          ← Ask something else
        </button>
      </div>

      {(result.transcript || result.doc_text) && (
        <section className="flex flex-col gap-2">
          {result.transcript && (
            <div className="rounded-xl bg-blue-50 p-3 text-sm text-slate-700">
              <span className="font-semibold">📝 We heard: </span>
              {result.transcript}
            </div>
          )}
          {result.doc_text && (
            <Collapsible title="📄 Text extracted from your document(s)">
              <pre className="whitespace-pre-wrap font-sans">{result.doc_text}</pre>
            </Collapsible>
          )}
        </section>
      )}

      {result.supported === false && (
        <div className="rounded-xl bg-amber-50 p-3 text-sm text-amber-900">
          This kind of legal problem isn't fully supported yet, so we didn't guess — a free
          legal aid lawyer below can help directly.
        </div>
      )}

      <SeverityBanner
        severity={result.severity}
        deadlineDays={result.deadline_days}
        vertical={result.vertical}
        qualifiesForAid={reasons.length > 0}
      />

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          2 · What the law says
        </h2>
        <p className="whitespace-pre-wrap text-base leading-relaxed text-slate-800">{answer}</p>
      </section>

      {audioSrc && (
        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
            🔊 Spoken answer
          </h2>
          <audio controls src={audioSrc} className="w-full" />
        </section>
      )}

      {reasons.length > 0 && (
        <section className="rounded-xl bg-green-50 p-4 text-sm text-green-900">
          <div className="mb-1 font-semibold">You likely qualify for FREE legal aid (Section 12):</div>
          <ul className="list-inside list-disc">
            {reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </section>
      )}

      {dlsa && (
        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
            3 · Talk to a real lawyer — free
          </h2>
          <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
            <div className="font-semibold">{dlsa.name}</div>
            {dlsa.phone && <div>📞 {dlsa.phone}</div>}
            {dlsa.hours && <div>🕑 {dlsa.hours}</div>}
            {dlsa.bring && <div>🎒 Bring: {dlsa.bring}</div>}
            {dlsa.tele_law && <div>💻 {dlsa.tele_law}</div>}
          </div>
        </section>
      )}

      {(result.citations?.length ?? 0) > 0 && (
        <Collapsible title="Cited statute sections (grounding)">
          <ul className="flex flex-col gap-1">
            {result.citations!.map((c, i) => (
              <li key={i}>
                <span className="font-semibold">
                  {c.act}, Section {c.section_no}
                </span>{" "}
                — {c.title}
              </li>
            ))}
          </ul>
          {result.ungrounded_claims && result.ungrounded_claims.length > 0 && (
            <p className="mt-2 text-xs text-slate-400">
              Stripped ungrounded claims: {result.ungrounded_claims.join(", ")}
            </p>
          )}
        </Collapsible>
      )}

      {result.error && result.error.length > 0 && (
        <div className="rounded-lg bg-amber-50 p-2.5 text-xs text-amber-700">
          Notes: {result.error.join(" | ")}
        </div>
      )}

      {(result.agent_trace?.length ?? 0) > 0 && (
        <Collapsible title="🔧 How we found this answer (technical details)">
          <AgentTrace agentTrace={result.agent_trace ?? []} signalPacket={result.signal_packet} />
        </Collapsible>
      )}
    </div>
  );
}
