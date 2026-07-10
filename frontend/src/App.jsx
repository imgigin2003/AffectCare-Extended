import { useEffect, useRef, useState } from 'react';
import { useTheme } from './hooks/useTheme';
import { useRecorder } from './hooks/useRecorder';
import { predictAudio } from './utils/api';
import ThemeToggle from './components/ThemeToggle';
import ResultCard from './components/ResultCard';
import ModelInsights from './components/ModelInsights';
import './App.css';

export default function App() {
  const { theme, toggle } = useTheme();
  const recorder = useRecorder();

  const [tab, setTab] = useState('detect'); // detect | insights
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [sourceName, setSourceName] = useState('');
  const [status, setStatus] = useState('idle'); // idle | analyzing | done | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const fileInputRef = useRef(null);

  // On refresh the browser drops in-memory blobs automatically — nothing
  // persists to disk here, and the backend deletes its copy after scoring.
  // We just revoke object URLs when they're replaced or the app unmounts.
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const resetResult = () => {
    setStatus('idle');
    setResult(null);
    setErrorMsg('');
  };

  const setClip = (blob, name) => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioBlob(blob);
    setAudioUrl(URL.createObjectURL(blob));
    setSourceName(name);
    resetResult();
  };

  const handleStartRecording = () => {
    resetResult();
    recorder.start();
  };

  const handleStopRecording = () => {
    try {
      const wav = recorder.stop();
      if (!wav) {
        setErrorMsg('Recording was empty — try again and speak into the mic.');
        setStatus('error');
        return;
      }
      setClip(wav, `recording-${new Date().toLocaleTimeString()}.wav`);
    } catch {
      setErrorMsg('Could not process the recording.');
      setStatus('error');
    }
  };

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setClip(file, file.name);
  };

  const handleAnalyze = async () => {
    if (!audioBlob) return;
    setStatus('analyzing');
    setErrorMsg('');
    try {
      const filename =
        sourceName.endsWith('.wav') || sourceName.endsWith('.mp3') ? sourceName : 'clip.wav';
      const res = await predictAudio(audioBlob, filename);
      setResult(res);
      setStatus('done');
    } catch (err) {
      setErrorMsg(err.message);
      setStatus('error');
    }
  };

  const clearClip = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioBlob(null);
    setAudioUrl(null);
    setSourceName('');
    resetResult();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const mm = String(Math.floor(recorder.seconds / 60)).padStart(2, '0');
  const ss = String(recorder.seconds % 60).padStart(2, '0');

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden>◈</span>
          <div>
            <h1>AffectCare</h1>
            <p>Vocal distress detection</p>
          </div>
        </div>
        <ThemeToggle theme={theme} onToggle={toggle} />
      </header>

      <nav className="tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === 'detect'}
          className={`tab ${tab === 'detect' ? 'tab-active' : ''}`}
          onClick={() => setTab('detect')}
        >
          Detect
        </button>
        <button
          role="tab"
          aria-selected={tab === 'insights'}
          className={`tab ${tab === 'insights' ? 'tab-active' : ''}`}
          onClick={() => setTab('insights')}
        >
          Model Insights
        </button>
      </nav>

      {tab === 'insights' ? (
        <main className="panel">
          <ModelInsights audioBlob={audioBlob} sourceName={sourceName} />
        </main>
      ) : (
      <main className="panel panel-detect">
        <p className="tagline">
          A distress signal shouldn't need a button. Record or upload a short clip and let the model listen.
        </p>

        <div className="actions">
          {!recorder.isRecording ? (
            <button className="btn btn-record" onClick={handleStartRecording}>
              <span className="dot" /> Record audio
            </button>
          ) : (
            <button className="btn btn-stop" onClick={handleStopRecording}>
              <span className="pulse" /> Stop · {mm}:{ss}
            </button>
          )}

          <span className="or">or</span>

          <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
            Upload .wav / .mp3
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,audio/wav,audio/mpeg"
            onChange={handleFile}
            hidden
          />
        </div>

        {recorder.error && <p className="hint hint-error">{recorder.error}</p>}

        {audioUrl && (
          <div className="clip">
            <div className="clip-head">
              <span className="clip-name" title={sourceName}>{sourceName}</span>
              <button className="link-btn" onClick={clearClip}>Clear</button>
            </div>
            <audio controls src={audioUrl} className="player" />
            <button className="btn btn-analyze" onClick={handleAnalyze} disabled={status === 'analyzing'}>
              {status === 'analyzing' ? 'Analyzing…' : 'Analyze clip'}
            </button>
          </div>
        )}

        {status === 'error' && <p className="hint hint-error">{errorMsg}</p>}

        {status === 'done' && result && <ResultCard result={result} />}
      </main>
      )}

      <footer className="footer">
        <p>Recall-first model · threshold 0.15 · clips are deleted after scoring, nothing is stored.</p>
      </footer>
    </div>
  );
}
