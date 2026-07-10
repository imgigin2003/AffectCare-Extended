import { useRef, useState, useCallback } from 'react';
import { encodeWav } from '../utils/wav';

// Records microphone audio as raw PCM via the Web Audio API and returns a
// ready-to-send WAV Blob on stop. We capture PCM directly (instead of using
// MediaRecorder's WebM/Opus output) so there's no fragile decode step and the
// result is always a WAV that librosa can read.
export function useRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState(null);

  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const sampleRateRef = useRef(44100);
  const timerRef = useRef(null);

  const cleanup = useCallback(() => {
    clearInterval(timerRef.current);
    try {
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close();
      }
    } catch {
      /* no-op */
    }
    processorRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;
      sampleRateRef.current = audioCtx.sampleRate;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      sourceRef.current = source;
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        // Copy — the underlying buffer is reused on the next callback
        chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setSeconds(0);
      setIsRecording(true);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch {
      setError('Microphone access denied or unavailable');
      cleanup();
    }
  }, [cleanup]);

  const stop = useCallback(() => {
    const sampleRate = sampleRateRef.current;
    const chunks = chunksRef.current;
    cleanup();
    setIsRecording(false);

    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    if (total === 0) return null;

    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    chunksRef.current = [];

    return encodeWav(merged, sampleRate);
  }, [cleanup]);

  return { isRecording, seconds, error, start, stop };
}
