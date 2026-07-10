export default function ResultCard({ result }) {
  const isDistress = result.label === 'distress';

  // The model returns P(distress). Confidence in the *shown verdict* is that
  // probability when distress, or its complement when normal — so a clip the
  // model is sure is normal reads as high confidence, not a misleading low %.
  const distressProb = result.confidence;
  const pct = Math.round((isDistress ? distressProb : 1 - distressProb) * 100);

  return (
    <div className={`result ${isDistress ? 'result-distress' : 'result-safe'}`}>
      <div className="result-top">
        <span className="result-icon" aria-hidden>{isDistress ? '⚠' : '✓'}</span>
        <div>
          <span className="result-label">{isDistress ? 'Distress detected' : 'No distress'}</span>
          <span className="result-sub">
            Model prediction: <strong>{result.label}</strong>
          </span>
        </div>
      </div>

      <div className="meter">
        <div className="meter-head">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="meter-track">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <p className="result-note">
        Alert threshold is {Math.round(result.threshold * 100)}% — this model is tuned to catch
        emergencies even at the cost of occasional false alarms.
      </p>
    </div>
  );
}
