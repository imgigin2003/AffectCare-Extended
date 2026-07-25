// Renders a (13 × 173) MFCC array to a base64 PNG using a magma-style colormap,
// mirroring the matplotlib `cmap="magma"` image the Python backend used to
// return — so the Insights tab renders identically, but client-side.

const MAGMA = [
  [0.0, [0, 0, 4]],
  [0.13, [28, 16, 68]],
  [0.25, [79, 18, 123]],
  [0.38, [129, 37, 129]],
  [0.5, [181, 54, 122]],
  [0.63, [229, 80, 100]],
  [0.75, [251, 135, 97]],
  [0.88, [254, 194, 135]],
  [1.0, [252, 253, 191]],
];

function magma(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < MAGMA.length; i++) {
    if (t <= MAGMA[i][0]) {
      const [t0, c0] = MAGMA[i - 1];
      const [t1, c1] = MAGMA[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return MAGMA[MAGMA.length - 1][1];
}

// Returns the raw base64 (no data: prefix) so callers can build a data URL.
export function renderSpectrogram(mfcc, min, max) {
  const nMfcc = mfcc.length; // 13
  const nFrames = mfcc[0].length; // 173
  const cellW = 4;
  const cellH = 10;

  const canvas = document.createElement('canvas');
  canvas.width = nFrames * cellW;
  canvas.height = nMfcc * cellH;
  const g = canvas.getContext('2d');
  const span = max - min || 1;

  for (let f = 0; f < nMfcc; f++) {
    for (let t = 0; t < nFrames; t++) {
      const [r, gr, b] = magma((mfcc[f][t] - min) / span);
      g.fillStyle = `rgb(${r},${gr},${b})`;
      // origin lower: band 0 at the bottom
      g.fillRect(t * cellW, (nMfcc - 1 - f) * cellH, cellW, cellH);
    }
  }

  return canvas.toDataURL('image/png').split(',')[1];
}
