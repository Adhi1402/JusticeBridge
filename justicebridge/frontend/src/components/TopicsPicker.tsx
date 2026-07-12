import { useEffect, useState } from "react";
import { getKbStores } from "../lib/api";
import type { KbStoresResponse } from "../lib/types";

interface Props {
  onPickTopic: (topic: string) => void;
}

/**
 * "What can I ask about?" chip list, sourced live from /kb-stores so the
 * frontend never hard-codes the vertical catalogue. Tapping a supported
 * topic seeds the textarea with a starter sentence; "coming soon" topics
 * are shown but disabled so users know what's on the roadmap without
 * being able to submit a query the backend can't yet handle.
 */
export default function TopicsPicker({ onPickTopic }: Props) {
  const [stores, setStores] = useState<KbStoresResponse | null>(null);

  useEffect(() => {
    getKbStores()
      .then(setStores)
      .catch(() => setStores(null));
  }, []);

  if (!stores) return null;

  const supported = Object.entries(stores.supported).filter(([, s]) => !s.cross_cutting);
  const comingSoon = Object.entries(stores.coming_soon);

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
        What can I ask about?
      </h2>
      <div className="flex flex-wrap gap-2">
        {supported.map(([id, info]) => (
          <button
            key={id}
            type="button"
            onClick={() => onPickTopic(info.topic)}
            title={info.description}
            className="rounded-full border border-[#1e3a5f]/30 bg-white px-3 py-1.5 text-sm font-medium text-[#1e3a5f] shadow-sm active:bg-blue-50"
          >
            {info.topic}
          </button>
        ))}
        {comingSoon.map(([id, info]) => (
          <span
            key={id}
            className="rounded-full border border-dashed border-slate-300 px-3 py-1.5 text-sm text-slate-400"
          >
            {info.topic} · soon
          </span>
        ))}
      </div>
    </section>
  );
}
