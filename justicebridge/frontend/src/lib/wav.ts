// The browser's MediaRecorder produces audio/webm (Opus), but the backend
// (agents/io_agents.py) writes recorded bytes straight to a *.wav file and
// hands it to Sarvam STT / Whisper with no transcoding step. Opus-in-WebM
// mislabeled as WAV either fails outright or transcribes badly, which
// silently degrades every downstream agent (planner routes off_topic, so
// escalation skips citations/DLSA/eligibility entirely). Fix: decode the
// recording and re-encode it as real PCM16 WAV before it ever leaves the
// browser, so the backend always receives what its filename claims.

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function floatTo16BitPCM(view: DataView, offset: number, input: Float32Array) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function interleave(left: Float32Array, right: Float32Array): Float32Array {
  const out = new Float32Array(left.length + right.length);
  for (let i = 0, j = 0; i < left.length; i++) {
    out[j++] = left[i];
    out[j++] = right[i];
  }
  return out;
}

function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitDepth = 16;

  const samples =
    numChannels === 2
      ? interleave(buffer.getChannelData(0), buffer.getChannelData(1))
      : buffer.getChannelData(0);

  const dataLength = samples.length * (bitDepth / 8);
  const arrayBuffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(arrayBuffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataLength, true);
  floatTo16BitPCM(view, 44, samples);

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

/** Decode a recorded blob (webm/ogg/whatever the browser produced) and
 * re-encode it as a real PCM16 WAV blob the backend can trust by extension. */
export async function transcodeToWav(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextCtor =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioContextCtor();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    return audioBufferToWavBlob(audioBuffer);
  } finally {
    void ctx.close();
  }
}
