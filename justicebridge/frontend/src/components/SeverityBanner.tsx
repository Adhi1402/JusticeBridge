import type { Severity } from "../lib/types";

const SEV: Record<Severity, { color: string; icon: string; label: string }> = {
  red: { color: "bg-sev-red", icon: "🔴", label: "Act now" },
  amber: { color: "bg-sev-amber", icon: "🟠", label: "Act soon" },
  green: { color: "bg-sev-green", icon: "🟢", label: "For your awareness" },
};

interface Props {
  severity?: Severity;
  deadlineDays?: number | null;
  vertical?: string | null;
  qualifiesForAid: boolean;
}

export default function SeverityBanner({ severity, deadlineDays, vertical, qualifiesForAid }: Props) {
  const sev = SEV[severity ?? "green"];
  const deadlineTxt = deadlineDays
    ? ` · act within ~${Math.max(1, Math.round(deadlineDays / 7))} weeks`
    : "";
  const sub =
    (vertical ? vertical.charAt(0).toUpperCase() + vertical.slice(1) : "") +
    (qualifiesForAid ? " · you likely qualify for FREE legal aid" : "");

  return (
    <div className={`${sev.color} rounded-2xl p-5 text-center text-white shadow-md`}>
      <div className="text-xl font-bold">
        {sev.icon} {sev.label}
        {deadlineTxt}
      </div>
      {sub && <div className="mt-1 text-sm font-normal opacity-95">{sub}</div>}
    </div>
  );
}
