import { useEffect, useRef } from 'react';

// Draws the clip's waveform on a canvas by decoding the audio blob client-side.
// Our clips are WAV (recorded) or WAV/MP3 (uploaded), all of which
// decodeAudioData handles — no server round-trip needed.
export default function Waveform({ blob }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const canvas = canvasRef.current;
      if (!canvas || !blob) return;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      try {
        const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
        if (cancelled) return;

        const data = decoded.getChannelData(0);
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.clientWidth;
        const cssH = canvas.clientHeight;
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;

        const g = canvas.getContext('2d');
        g.scale(dpr, dpr);
        g.clearRect(0, 0, cssW, cssH);

        const accent =
          getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#4f7cff';
        const mid = cssH / 2;
        const barW = 2;
        const gap = 1;
        const step = Math.floor(data.length / (cssW / (barW + gap))) || 1;

        g.fillStyle = accent;
        for (let x = 0, i = 0; x < cssW; x += barW + gap, i += step) {
          let min = 1;
          let max = -1;
          for (let j = 0; j < step && i + j < data.length; j++) {
            const s = data[i + j];
            if (s < min) min = s;
            if (s > max) max = s;
          }
          const h = Math.max(1, ((max - min) / 2) * cssH * 0.9);
          g.fillRect(x, mid - h, barW, h * 2);
        }
      } catch {
        const g = canvas.getContext('2d');
        g.clearRect(0, 0, canvas.width, canvas.height);
      } finally {
        ctx.close();
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [blob]);

  return <canvas ref={canvasRef} className="waveform" />;
}
