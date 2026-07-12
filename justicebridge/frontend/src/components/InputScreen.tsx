import { useEffect, useRef, useState } from "react";
import type { Lang } from "../lib/types";
import { transcodeToWav } from "../lib/wav";
import LanguageSelector from "./LanguageSelector";
import TopicsPicker from "./TopicsPicker";
import Logo from "./Logo";

export interface InputSubmission {
  text: string;
  audioBlob: Blob | null;
  images: File[];
}

interface PickedImage {
  file: File;
  url: string;
}

interface Props {
  lang: Lang;
  onChangeLang: (lang: Lang) => void;
  onSubmit: (data: InputSubmission) => void;
  busy: boolean;
}

type RecordState = "idle" | "recording" | "processing" | "recorded";

/**
 * Screen 1 — the entry point of the whole pipeline: this is where the
 * ASR + Vision agents (graph.py's parallel START branches) get their raw
 * input. Everything here is optional individually, but at least one of
 * text / voice / document is required before we can call /ask.
 */
export default function InputScreen({ lang, onChangeLang, onSubmit, busy }: Props) {
  const [text, setText] = useState("");
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [images, setImages] = useState<PickedImage[]>([]);
  const [micError, setMicError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const hasInput = text.trim().length > 0 || audioBlob !== null || images.length > 0;

  // Revoke every object URL on unmount so we don't leak blob memory.
  useEffect(() => {
    return () => {
      images.forEach((img) => URL.revokeObjectURL(img.url));
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startRecording() {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const rawBlob = new Blob(chunksRef.current, { type: recorder.mimeType });
        setRecordState("processing");
        // The recorder outputs webm/Opus; re-encode to real PCM16 WAV so the
        // backend's ASR (Sarvam / Whisper) — which trusts the .wav it's
        // handed — actually gets a WAV file instead of mislabeled Opus. If
        // this fails, DO NOT silently send the raw (broken) blob — that
        // reproduces the exact "off-topic, no citations" failure with only
        // an easy-to-miss warning. Force a re-record instead.
        try {
          const wavBlob = await transcodeToWav(rawBlob);
          setAudioBlob(wavBlob);
          setAudioUrl(URL.createObjectURL(wavBlob));
          setRecordState("recorded");
        } catch (e) {
          console.error("transcodeToWav failed:", e);
          setMicError(
            "Couldn't process that recording in this browser. Please try recording again, " +
              "or type your question instead."
          );
          setRecordState("idle");
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordState("recording");
    } catch {
      setMicError("Couldn't access the microphone. Check browser permissions.");
    }
  }

  function stopRecording() {
    // recorder.onstop (async) transcodes to WAV, then sets state to "recorded".
    mediaRecorderRef.current?.stop();
  }

  function discardRecording() {
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setRecordState("idle");
  }

  function onFilesPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []).map((file) => ({
      file,
      url: URL.createObjectURL(file),
    }));
    setImages((prev) => [...prev, ...picked]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function removeImage(idx: number) {
    setImages((prev) => {
      URL.revokeObjectURL(prev[idx].url);
      return prev.filter((_, i) => i !== idx);
    });
  }

  function handleSubmit() {
    if (!hasInput || busy) return;
    onSubmit({ text: text.trim(), audioBlob, images: images.map((i) => i.file) });
  }

  function seedFromTopic(topic: string) {
    if (text.trim()) return; // don't clobber something the user already typed
    setText(`I have a problem about ${topic.toLowerCase()}: `);
    textareaRef.current?.focus();
  }

  return (
    <div className="flex flex-col gap-6 px-4 pb-28 pt-4">
      <header className="flex flex-col items-center gap-1 pt-2 text-center">
        <Logo />
      </header>

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Language
        </h2>
        <LanguageSelector value={lang} onChange={onChangeLang} />
      </section>

      <TopicsPicker onPickTopic={seedFromTopic} />

      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          1 · Tell us what happened
        </h2>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. I worked two months and my contractor hasn't paid my wages…"
          rows={4}
          className="w-full rounded-xl border border-slate-300 bg-white p-3 text-base shadow-sm focus:border-[#1e3a5f] focus:outline-none focus:ring-1 focus:ring-[#1e3a5f]"
        />
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          …or speak
        </h2>
        {recordState === "idle" && (
          <button
            type="button"
            onClick={startRecording}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white py-4 text-base font-medium shadow-sm active:bg-slate-100"
          >
            🎙️ Tap to record
          </button>
        )}
        {recordState === "recording" && (
          <button
            type="button"
            onClick={stopRecording}
            className="flex w-full animate-pulse items-center justify-center gap-2 rounded-xl bg-red-600 py-4 text-base font-medium text-white shadow-sm"
          >
            ⏹ Recording… tap to stop
          </button>
        )}
        {recordState === "processing" && (
          <div className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white py-4 text-base font-medium text-slate-500 shadow-sm">
            Processing recording…
          </div>
        )}
        {recordState === "recorded" && audioUrl && (
          <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-white p-3 shadow-sm">
            <audio controls src={audioUrl} className="w-full" />
            <button
              type="button"
              onClick={discardRecording}
              className="self-start text-sm text-red-600 underline"
            >
              Discard and re-record
            </button>
          </div>
        )}
        {micError && <p className="text-sm text-red-600">{micError}</p>}
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          …or scan paper(s)
        </h2>
        <p className="text-xs text-slate-500">Notice / contract / bill</p>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white py-4 text-base font-medium shadow-sm active:bg-slate-100"
        >
          📷 Add photo
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          multiple
          className="hidden"
          onChange={onFilesPicked}
        />
        {images.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {images.map((img, i) => (
              <div key={img.url} className="relative">
                <img
                  src={img.url}
                  alt={img.file.name}
                  className="h-20 w-20 rounded-lg object-cover"
                />
                <button
                  type="button"
                  onClick={() => removeImage(i)}
                  aria-label={`Remove ${img.file.name}`}
                  className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-red-600 text-xs text-white shadow"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      <div className="fixed inset-x-0 bottom-0 border-t border-slate-200 bg-white/95 p-4 backdrop-blur">
        <button
          type="button"
          disabled={!hasInput || busy}
          onClick={handleSubmit}
          className="mx-auto flex w-full max-w-md items-center justify-center gap-2 rounded-xl bg-[#1e3a5f] py-4 text-lg font-semibold text-white shadow-md disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? "Working…" : "Get help"}
        </button>
      </div>
    </div>
  );
}
