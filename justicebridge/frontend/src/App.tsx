import { useState } from "react";
import InputScreen, { type InputSubmission } from "./components/InputScreen";
import ResultScreen from "./components/ResultScreen";
import StatusBar from "./components/StatusBar";
import LiveProgress from "./components/LiveProgress";
import { askStream, blobToBase64 } from "./lib/api";
import type { AgentTraceEntry, AskResponse, Lang } from "./lib/types";

type Screen = "input" | "result";

export default function App() {
  const [screen, setScreen] = useState<Screen>("input");
  const [lang, setLang] = useState<Lang>("en");
  const [busy, setBusy] = useState(false);
  const [liveSteps, setLiveSteps] = useState<AgentTraceEntry[]>([]);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit({ text, audioBlob, images }: InputSubmission) {
    setBusy(true);
    setError(null);
    setLiveSteps([]);
    try {
      const audio_base64 = audioBlob ? await blobToBase64(audioBlob) : undefined;
      const images_base64 = images.length ? await Promise.all(images.map(blobToBase64)) : undefined;

      // Streams one agent's result the moment it finishes, instead of one
      // 20s-2min blocking request — see LiveProgress for how these render.
      const response = await askStream(
        {
          lang,
          want_tts: true,
          text_input: text || undefined,
          audio_base64,
          images_base64,
        },
        (step) => setLiveSteps((prev) => [...prev, step])
      );

      setResult(response);
      setScreen("result");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  function startOver() {
    setResult(null);
    setError(null);
    setLiveSteps([]);
    setScreen("input");
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col bg-slate-50">
      <StatusBar />
      {error && (
        <div className="m-4 rounded-lg bg-red-50 p-3 text-sm text-red-700" role="alert">
          {error}
        </div>
      )}
      {busy && <LiveProgress steps={liveSteps} />}
      {screen === "input" && !busy && (
        <InputScreen lang={lang} onChangeLang={setLang} onSubmit={handleSubmit} busy={busy} />
      )}
      {screen === "result" && result && <ResultScreen result={result} onStartOver={startOver} />}
    </div>
  );
}
