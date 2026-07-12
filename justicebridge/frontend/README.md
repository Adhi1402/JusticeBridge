# JusticeBridge frontend

Mobile-first, single-column web app (React + TypeScript + Vite + Tailwind)
against the FastAPI backend in [`../api.py`](../api.py).

The UI mirrors the pipeline in `graph.py`: the first screen is the input
step (text / voice / document — the ASR + Vision agents' raw input), and
submitting it runs the whole graph in one call to `POST /ask`, landing on a
result screen (severity signal, plain-language answer, spoken answer,
citations, and the free-legal-aid handoff).

## Run

Backend, in one terminal (from the repo root):
```bash
pip install fastapi uvicorn
uvicorn justicebridge.api:app --host 0.0.0.0 --port 8080
```

Frontend, in another terminal:
```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8080` in dev (see
`vite.config.ts`), so the browser never needs a second CORS-enabled origin.
For a production build pointed at a different backend host, set
`VITE_API_BASE_URL` (e.g. in `.env.local`) to the full API base URL.

## Structure

- `src/lib/types.ts` — mirrors `api.py`'s `AskRequest` / `_RESPONSE_FIELDS` contract.
- `src/lib/api.ts` — thin fetch client (`getHealth`, `getKbStores`, `ask`).
- `src/components/InputScreen.tsx` — text / mic recording / camera capture, the pipeline's entry point.
- `src/components/ResultScreen.tsx` — severity banner, answer, TTS playback, citations, DLSA contact.
- `src/components/StatusBar.tsx` — backend health strip.

## Notes

- Voice recording uses the browser `MediaRecorder` API, which typically
  produces `audio/webm`, not WAV. The backend ASR agents (Sarvam / Whisper)
  generally handle common container formats, but if you hit transcription
  issues, check `asr_confidence` in the result and consider transcoding
  client-side before base64-encoding.
- Camera capture uses `<input type="file" capture="environment">`, which
  opens the rear camera directly on mobile browsers that support it.
