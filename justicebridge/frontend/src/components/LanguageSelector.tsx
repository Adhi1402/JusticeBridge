import type { Lang } from "../lib/types";

const LANGS: { value: Lang; label: string }[] = [
  { value: "en", label: "English" },
  { value: "ta", label: "தமிழ்" },
  { value: "hi", label: "हिन्दी" },
  { value: "te", label: "తెలుగు" },
];

interface Props {
  value: Lang;
  onChange: (lang: Lang) => void;
}

export default function LanguageSelector({ value, onChange }: Props) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1" role="radiogroup" aria-label="Language">
      {LANGS.map((l) => (
        <button
          key={l.value}
          type="button"
          role="radio"
          aria-checked={value === l.value}
          onClick={() => onChange(l.value)}
          className={`shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
            value === l.value
              ? "bg-[#1e3a5f] text-white"
              : "bg-white text-slate-600 border border-slate-300"
          }`}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}
