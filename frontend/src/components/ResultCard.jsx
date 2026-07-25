export default function ResultCard({ result }) {
  // Verdict follows the majority class (≥40% distress → distress), matching
  // predict.py. The confidence shown is relative to that verdict, so a clip the
  // model is sure is normal reads as high confidence, not a misleading low %.
  const distressProb = result.confidence;
  const isDistress = distressProb >= 0.4;
  const pct = Math.round((isDistress ? distressProb : 1 - distressProb) * 100);

  return (
    <div className={`result ${isDistress ? "result-distress" : "result-safe"}`}>
      <div className="result-top">
        <span className="result-icon" aria-hidden>
          {isDistress ? "⚠" : "✓"}
        </span>
        <div>
          <span className="result-label">
            {isDistress ? "Distress detected" : "No distress"}
          </span>
          <span className="result-sub">
            <strong>{Math.round(distressProb * 100)}%</strong> distress
            probability
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
        Alert threshold is {Math.round(result.threshold * 100)}% — this model is
        tuned to catch emergencies even at the cost of occasional false alarms.
      </p>
    </div>
  );
}
