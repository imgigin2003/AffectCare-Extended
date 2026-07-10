import { useEffect, useState } from 'react';
import { getMetrics, getLossHistory, visualizeAudio } from '../utils/api';
import ConfusionMatrix from './ConfusionMatrix';
import LossCurve from './LossCurve';
import Waveform from './Waveform';

export default function ModelInsights({ audioBlob, sourceName }) {
  const [metrics, setMetrics] = useState(null);
  const [loss, setLoss] = useState(null);
  const [modelError, setModelError] = useState('');

  const [viz, setViz] = useState(null);
  const [vizState, setVizState] = useState('idle'); // idle | loading | done | error
  const [vizError, setVizError] = useState('');

  // Model-level data — loaded once
  useEffect(() => {
    Promise.all([getMetrics(), getLossHistory()])
      .then(([m, l]) => {
        setMetrics(m);
        setLoss(l);
      })
      .catch((e) => setModelError(e.message));
  }, []);

  // Per-clip spectrogram — recomputed whenever the clip changes
  useEffect(() => {
    if (!audioBlob) {
      setViz(null);
      setVizState('idle');
      return;
    }
    let cancelled = false;
    setVizState('loading');
    setVizError('');
    visualizeAudio(audioBlob, sourceName || 'clip.wav')
      .then((res) => {
        if (cancelled) return;
        setViz(res);
        setVizState('done');
      })
      .catch((e) => {
        if (cancelled) return;
        setVizError(e.message);
        setVizState('error');
      });
    return () => {
      cancelled = true;
    };
  }, [audioBlob, sourceName]);

  return (
    <div className="insights">
      {/* ---- Model-level performance ---- */}
      <section>
        <h2 className="section-title">Model performance</h2>
        <p className="section-sub">Measured on the held-out 20% test split — the same model every clip runs through.</p>

        {modelError && <p className="hint hint-error">{modelError}</p>}

        {metrics && (
          <>
            <div className="tiles">
              <Tile label="Recall" value={pct(metrics.recall)} note="emergencies caught" tone="accent" />
              <Tile label="Precision" value={pct(metrics.precision)} note="alarms that were real" />
              <Tile label="F1 score" value={pct(metrics.f1)} note="recall + precision balance" />
            </div>

            <div className="viz-grid">
              <figure className="viz-card">
                <figcaption>Confusion matrix · {metrics.test_size} test clips</figcaption>
                <ConfusionMatrix cm={metrics.confusion_matrix} />
              </figure>
              <figure className="viz-card">
                <figcaption>Training loss · {loss ? loss.length : 0} epochs</figcaption>
                {loss && <LossCurve loss={loss} />}
              </figure>
            </div>
          </>
        )}
      </section>

      {/* ---- Per-clip visualization ---- */}
      <section>
        <h2 className="section-title">This clip</h2>
        <p className="section-sub">What the current audio looks like — and the exact MFCC image the CNN reads.</p>

        {!audioBlob && (
          <p className="empty">Record or upload a clip in the Detect tab to visualize it here.</p>
        )}

        {audioBlob && (
          <div className="viz-grid">
            <figure className="viz-card">
              <figcaption>Waveform · {sourceName}</figcaption>
              <Waveform blob={audioBlob} />
            </figure>
            <figure className="viz-card">
              <figcaption>MFCC spectrogram (13 bands × time)</figcaption>
              {vizState === 'loading' && <div className="viz-skeleton">Computing…</div>}
              {vizState === 'error' && <p className="hint hint-error">{vizError}</p>}
              {vizState === 'done' && viz && (
                <>
                  <img
                    className="spectrogram"
                    src={`data:image/png;base64,${viz.mfcc_png}`}
                    alt="MFCC spectrogram of the uploaded clip"
                  />
                  <div className="viz-axis">
                    <span>← time →</span>
                    <span>energy {viz.stats.energy_min} to {viz.stats.energy_max}</span>
                  </div>
                </>
              )}
            </figure>
          </div>
        )}
      </section>
    </div>
  );
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

function Tile({ label, value, note, tone }) {
  return (
    <div className={`tile ${tone === 'accent' ? 'tile-accent' : ''}`}>
      <span className="tile-value">{value}</span>
      <span className="tile-label">{label}</span>
      <span className="tile-note">{note}</span>
    </div>
  );
}
