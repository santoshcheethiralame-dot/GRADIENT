import { useCallback, useEffect, useRef, useState } from 'react';
import { getGpuContext, WebGpuUnsupportedError } from './gpu/device';
import {
  runSelfTest,
  type SelfTestReport,
  type CheckResult,
  type CheckGroup,
  type TrainingResult,
} from './gpu/selftest';
import Dashboard from './ui/Dashboard';

type Phase =
  | { kind: 'init' }
  | { kind: 'ready'; label: string; info: GPUAdapterInfo }
  | { kind: 'unsupported'; message: string }
  | { kind: 'error'; message: string };

const GROUP_LABELS: Record<CheckGroup, string> = {
  matmul: 'matmul kernels',
  forward: 'forward ops',
  composed: 'composed forward',
  backward: 'backward ops',
  optim: 'optimizers',
  data: 'data pipeline',
  gradcheck: 'gradient check · numerical vs. analytic',
};
const GROUP_ORDER: CheckGroup[] = [
  'matmul',
  'forward',
  'composed',
  'backward',
  'optim',
  'data',
  'gradcheck',
];

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: 'init' });
  const [report, setReport] = useState<SelfTestReport | null>(null);
  const [running, setRunning] = useState(false);
  const booted = useRef(false);

  const executeSelfTest = useCallback(async () => {
    setRunning(true);
    setReport(null);
    try {
      setReport(await runSelfTest());
    } catch (e) {
      setPhase({ kind: 'error', message: errMessage(e) });
    } finally {
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    if (booted.current) return; // guard double-invoke
    booted.current = true;
    (async () => {
      try {
        const ctx = await getGpuContext();
        setPhase({ kind: 'ready', label: ctx.label, info: ctx.info });
        await executeSelfTest();
      } catch (e) {
        if (e instanceof WebGpuUnsupportedError) {
          setPhase({ kind: 'unsupported', message: e.message });
        } else {
          setPhase({ kind: 'error', message: errMessage(e) });
        }
      }
    })();
  }, [executeSelfTest]);

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1 className="wordmark">
            gradient<span className="dot">.</span>
          </h1>
          <p className="tagline">
            A neural network that trains entirely on your GPU, in the browser — WebGPU compute
            shaders for matmul, the forward pass, backprop, and optimization.
          </p>
        </div>
        <StatusPill phase={phase} running={running} />
      </header>

      <div className="grid">
        <DeviceCard phase={phase} />

        {phase.kind === 'ready' && report?.training && <TrainingCard t={report.training} />}

        {phase.kind === 'ready' && <Dashboard />}

        {phase.kind === 'ready' && (
          <section className="card">
            <h2>Kernel self-test · GPU vs. CPU oracle</h2>
            <SelfTestView report={report} running={running} onRerun={executeSelfTest} />
          </section>
        )}

        {phase.kind === 'unsupported' && (
          <section className="card">
            <div className="banner fail">
              <span className="big">WebGPU unavailable</span>
            </div>
            <p className="tagline" style={{ marginTop: 14 }}>
              {phase.message}
            </p>
            <p className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12.5 }}>
              This is expected in some headless/CI browsers. Open in desktop Chrome or Edge 113+ to
              run the GPU kernels.
            </p>
          </section>
        )}

        {phase.kind === 'error' && (
          <section className="card">
            <div className="banner fail">
              <span className="big">Initialization error</span>
            </div>
            <p className="tagline" style={{ marginTop: 14 }}>
              <code className="inline">{phase.message}</code>
            </p>
          </section>
        )}
      </div>

      <p className="foot">
        All 6 phases complete — WebGPU device → tensors → matmul → forward → backprop (gradient-checked)
        → SGD/Adam → MNIST → a Worker-driven live dashboard with D3 heatmaps and draw-a-digit inference.
      </p>
    </div>
  );
}

function StatusPill({ phase, running }: { phase: Phase; running: boolean }) {
  let cls = 'pill';
  let text = '';
  if (phase.kind === 'init') {
    cls += ' busy';
    text = 'initializing webgpu…';
  } else if (phase.kind === 'ready') {
    if (running) {
      cls += ' busy';
      text = 'running kernels…';
    } else {
      cls += ' ok';
      text = 'gpu ready';
    }
  } else {
    cls += ' err';
    text = phase.kind === 'unsupported' ? 'unsupported' : 'error';
  }
  return (
    <span className={cls}>
      <span className="dot-led" />
      {text}
    </span>
  );
}

function DeviceCard({ phase }: { phase: Phase }) {
  return (
    <section className="card">
      <h2>Device</h2>
      <dl className="kv">
        <dt>status</dt>
        <dd>
          {phase.kind === 'init' && 'requesting adapter…'}
          {phase.kind === 'ready' && 'connected'}
          {phase.kind === 'unsupported' && 'unsupported'}
          {phase.kind === 'error' && 'error'}
        </dd>
        {phase.kind === 'ready' && (
          <>
            <dt>adapter</dt>
            <dd>{phase.label}</dd>
            <dt>vendor</dt>
            <dd>{phase.info.vendor || '—'}</dd>
            <dt>architecture</dt>
            <dd>{phase.info.architecture || '—'}</dd>
          </>
        )}
      </dl>
    </section>
  );
}

function SelfTestView({
  report,
  running,
  onRerun,
}: {
  report: SelfTestReport | null;
  running: boolean;
  onRerun: () => void;
}) {
  if (!report && running) {
    return <p className="banner info">Dispatching compute passes…</p>;
  }
  if (!report) {
    return <p className="banner info">Idle.</p>;
  }

  const passed = report.results.filter((r) => r.pass).length;
  const total = report.results.length;

  return (
    <>
      <div className={`banner ${report.allPassed ? 'pass' : 'fail'}`} style={{ marginBottom: 4 }}>
        <span className="big">{report.allPassed ? '✓ ALL PASS' : '✗ FAILURES'}</span>
        <span>
          {passed}/{total} checks within tolerance (rel. err &lt; 1e-3)
        </span>
      </div>

      {GROUP_ORDER.map((group) => {
        const rows = report.results.filter((r) => r.group === group);
        if (rows.length === 0) return null;
        return (
          <GroupTable
            key={group}
            title={GROUP_LABELS[group]}
            rows={rows}
            showThroughput={group === 'matmul'}
          />
        );
      })}

      <div className="row" style={{ marginTop: 18 }}>
        <button className="btn" onClick={onRerun} disabled={running}>
          {running ? 'running…' : 're-run self-test'}
        </button>
        <span className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
          round-trip includes readback; throughput is indicative, not a benchmark.
        </span>
      </div>
    </>
  );
}

function GroupTable({
  title,
  rows,
  showThroughput,
}: {
  title: string;
  rows: CheckResult[];
  showThroughput: boolean;
}) {
  return (
    <div style={{ marginTop: 18 }}>
      <div className="group-label">{title}</div>
      <table className="results">
        <thead>
          <tr>
            <th className="l">op</th>
            <th className="l">detail</th>
            <th>max abs err</th>
            <th>max rel err</th>
            <th>time</th>
            {showThroughput && <th>throughput</th>}
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="l">{r.name}</td>
              <td className="l detail">{r.detail}</td>
              <td>{r.maxAbsErr.toExponential(2)}</td>
              <td>{r.maxRelErr.toExponential(2)}</td>
              <td>{r.ms.toFixed(2)} ms</td>
              {showThroughput && <td>{r.throughput ?? '—'}</td>}
              <td className={`verdict ${r.pass ? 'pass' : 'fail'}`}>{r.pass ? 'PASS' : 'FAIL'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrainingCard({ t }: { t: TrainingResult }) {
  return (
    <section className="card">
      <h2>Training demo · Adam on synthetic data</h2>
      <div className={`banner ${t.pass ? 'pass' : 'fail'}`} style={{ marginBottom: 18 }}>
        <span className="big">{t.pass ? '✓ IT LEARNS' : '✗ NO CONVERGENCE'}</span>
        <span>
          loss {t.initialLoss.toFixed(2)} → {t.finalLoss.toFixed(3)} · accuracy{' '}
          {(t.accuracy * 100).toFixed(1)}%
        </span>
      </div>
      <div className="train-row">
        <div>
          <Sparkline data={t.lossHistory} />
          <div className="muted spark-cap">cross-entropy loss · {t.steps} full-batch steps</div>
        </div>
        <dl className="kv">
          <dt>architecture</dt>
          <dd>{t.arch}</dd>
          <dt>optimizer</dt>
          <dd>{t.optimizer}</dd>
          <dt>data</dt>
          <dd>{t.points} points</dd>
          <dt>final loss</dt>
          <dd>{t.finalLoss.toFixed(4)}</dd>
          <dt>wall time</dt>
          <dd>{t.ms.toFixed(0)} ms</dd>
        </dl>
      </div>
    </section>
  );
}

function Sparkline({ data, width = 360, height = 64 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const px = (i: number) => (i / (data.length - 1)) * width;
  const py = (v: number) => height - 3 - ((v - min) / range) * (height - 6);
  const line = data.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
  const area = `0,${height} ${line} ${width},${height}`;
  return (
    <svg width={width} height={height} className="spark" role="img" aria-label="training loss curve">
      <polygon points={area} fill="rgba(200,241,64,0.10)" />
      <polyline points={line} fill="none" stroke="var(--lime)" strokeWidth={2} strokeLinejoin="round" />
    </svg>
  );
}

function errMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
