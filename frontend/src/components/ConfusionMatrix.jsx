// Renders the 2x2 confusion matrix live from the model's real test counts.
// Rows = actual class, columns = predicted class. Diagonal = correct (green),
// off-diagonal = errors (red); every cell is labeled so the good/bad meaning
// never rests on color alone. Intensity scales with the count.
export default function ConfusionMatrix({ cm }) {
  const { true_negative: tn, false_positive: fp, false_negative: fn, true_positive: tp } = cm;
  const max = Math.max(tn, fp, fn, tp) || 1;

  // grid order: actual-normal row, then actual-distress row
  const cells = [
    { count: tn, correct: true, role: 'True negative', desc: 'normal → normal' },
    { count: fp, correct: false, role: 'False alarm', desc: 'normal → distress' },
    { count: fn, correct: false, role: 'Missed', desc: 'distress → normal' },
    { count: tp, correct: true, role: 'Caught', desc: 'distress → distress' },
  ];

  const bg = (c) => {
    const hue = c.correct ? 'var(--safe)' : 'var(--danger)';
    const strength = 12 + Math.round((c.count / max) * 48); // 12%–60%
    return `color-mix(in srgb, ${hue} ${strength}%, transparent)`;
  };

  return (
    <div className="cm">
      <div className="cm-corner" />
      <div className="cm-collabel">Predicted normal</div>
      <div className="cm-collabel">Predicted distress</div>

      <div className="cm-rowlabel">Actual normal</div>
      {cells.slice(0, 2).map((c) => (
        <Cell key={c.role} c={c} bg={bg(c)} />
      ))}

      <div className="cm-rowlabel">Actual distress</div>
      {cells.slice(2).map((c) => (
        <Cell key={c.role} c={c} bg={bg(c)} />
      ))}
    </div>
  );
}

function Cell({ c, bg }) {
  return (
    <div
      className={`cm-cell ${c.correct ? 'cm-correct' : 'cm-error'}`}
      style={{ background: bg }}
      title={`${c.role}: ${c.count} (${c.desc})`}
    >
      <span className="cm-count">{c.count}</span>
      <span className="cm-role">
        {c.correct ? '✓' : '✕'} {c.role}
      </span>
    </div>
  );
}
