interface Props {
  size?: "sm" | "lg";
}

// Single shared brand mark so every screen (input, loading, result) reads as
// the same professional product instead of a stack of separately-styled dev
// screens.
export default function Logo({ size = "lg" }: Props) {
  const badge = size === "lg" ? "h-12 w-12 text-2xl" : "h-8 w-8 text-base";
  const title = size === "lg" ? "text-xl" : "text-sm";

  return (
    <div className="flex items-center justify-center gap-2">
      <div
        className={`flex ${badge} shrink-0 items-center justify-center rounded-xl bg-[#1e3a5f] shadow-sm`}
        aria-hidden="true"
      >
        ⚖️
      </div>
      {size === "lg" && (
        <div className="flex flex-col items-start leading-tight">
          <span className={`${title} font-bold text-[#1e3a5f]`}>JusticeBridge</span>
          <span className="text-xs text-slate-500">Free legal help, in your language</span>
        </div>
      )}
    </div>
  );
}
