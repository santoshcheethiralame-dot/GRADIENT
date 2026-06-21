import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { getGpuContext, WebGpuUnsupportedError } from './gpu/device';
import {
  runSelfTest,
  type SelfTestReport,
  type CheckGroup,
  type TrainingResult,
} from './gpu/selftest';
import Dashboard from './ui/Dashboard';
import CpuTrainer from './ui/CpuTrainer';
import { NanoGptLab } from './ui/NanoGptLab';
import { ShakespeareLab } from './ui/ShakespeareLab';
import { RaceLab } from './ui/RaceLab';
import { About } from './ui/About';
import { runProfile, type ProfileRow } from './gpu/profile';

interface DeviceLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
}

type Phase =
  | { kind: 'init' }
  | { kind: 'ready'; label: string; info: GPUAdapterInfo; limits: DeviceLimits }
  | { kind: 'cpu' }
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
  nanogpt: 'nano-GPT · char transformer (forward · backprop · train)',
  tfgpu: 'nano-GPT · GPU kernels (forward + backward vs. f64 CPU)',
};
const GROUP_ORDER: CheckGroup[] = [
  'matmul',
  'forward',
  'composed',
  'backward',
  'optim',
  'data',
  'gradcheck',
  'nanogpt',
  'tfgpu',
];

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: 'init' });
  const [report, setReport] = useState<SelfTestReport | null>(null);
  const [running, setRunning] = useState(false);
  const [view, setView] = useState<'lab' | 'about'>('lab');
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
    if (booted.current) return;
    booted.current = true;
    if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('cpu')) {
      setPhase({ kind: 'cpu' });
      return;
    }
    (async () => {
      try {
        const ctx = await getGpuContext();
        const L = ctx.device.limits;
        setPhase({
          kind: 'ready',
          label: ctx.label,
          info: ctx.info,
          limits: {
            maxBufferSize: L.maxBufferSize,
            maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
            maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
          },
        });
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
        <div className="brand">
          <h1 className="wordmark">
            gradient<span className="dot" />
          </h1>

        </div>
        <div className="statline">
          <nav className="nav">
            <button className={view === 'lab' ? 'active' : ''} onClick={() => setView('lab')}>
              lab
            </button>
            <button className={view === 'about' ? 'active' : ''} onClick={() => setView('about')}>
              about
            </button>
          </nav>
          {phase.kind === 'ready' && (
            <span>
              GPU&nbsp;<b>{phase.label}</b>
            </span>
          )}
          {report && (
            <span className={report.allPassed ? 'nominal' : 'fault'}>
              {report.results.filter((r) => r.pass).length}/{report.results.length} KERNELS{' '}
              {report.allPassed ? 'NOMINAL' : 'FAULT'}
            </span>
          )}
          <StatusPill phase={phase} running={running} />
        </div>
      </header>

      {view === 'about' ? (
        <About onEnter={() => setView('lab')} />
      ) : (
        <>
          {phase.kind === 'ready' && <Overview phase={phase} report={report} />}

          <div className="grid">
        <DeviceCard phase={phase} />
        {phase.kind === 'ready' && report?.training && <TrainingCard t={report.training} />}

        {phase.kind === 'ready' && <Dashboard />}

        {phase.kind === 'ready' && (
          <section className="card" id="diag">
            <h2>
              <span className="ch">CH3</span>Kernel diagnostics
              <span className="meta">GPU vs. f64 oracle · every load</span>
            </h2>
            <SelfTestView report={report} running={running} onRerun={executeSelfTest} />
          </section>
        )}

        {phase.kind === 'ready' && <ProfilerCard />}

        {(phase.kind === 'ready' || phase.kind === 'cpu' || phase.kind === 'unsupported') && (
          <NanoGptLab />
        )}

        {phase.kind === 'ready' && <ShakespeareLab />}

        {phase.kind === 'ready' && <RaceLab />}

        {(phase.kind === 'cpu' || phase.kind === 'unsupported') && (
          <CpuTrainer forced={phase.kind === 'cpu'} />
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
            The full WebGPU stack — device → tensors → tiled matmul → forward → gradient-checked
            backprop → SGD/Adam → MNIST — plus nano-GPT, a char transformer trained on the GPU.
            49/49 self-test on every load.
          </p>
        </>
      )}
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
  } else if (phase.kind === 'cpu') {
    text = 'cpu mode';
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
  const specs: Array<[string, string]> = [];
  if (phase.kind === 'ready') {
    specs.push(
      ['status', 'connected'],
      ['adapter', phase.label],
      ['vendor', phase.info.vendor || '—'],
      ['architecture', phase.info.architecture || '—'],
      ['max buffer', fmtBytes(phase.limits.maxBufferSize)],
      ['max binding', fmtBytes(phase.limits.maxStorageBufferBindingSize)],
      ['wg invocations', phase.limits.maxComputeInvocationsPerWorkgroup.toLocaleString()],
      ['wg size · x', phase.limits.maxComputeWorkgroupSizeX.toLocaleString()],
      ['precision', 'f32'],
      ['memory', 'row-major'],
      ['shaders', 'WGSL'],
      ['queue', '1 · ordered'],
    );
  } else if (phase.kind === 'init') {
    specs.push(['status', 'requesting adapter…']);
  } else {
    specs.push(['status', phase.kind]);
  }
  return (
    <section className="card" id="device">
      <h2>
        <span className="ch">SYS</span>Device
      </h2>
      <div className="specs">
        {specs.map(([k, v]) => (
          <div className="spec" key={k}>
            <div className="spec-k">{k}</div>
            <div className="spec-v">{v}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Overview({ phase, report }: { phase: Phase; report: SelfTestReport | null }) {
  const arch = phase.kind === 'ready' ? phase.info.architecture || phase.info.vendor || 'gpu' : 'gpu';
  const passed = report ? report.results.filter((r) => r.pass).length : 0;
  const total = report ? report.results.length : 0;
  const cards: Array<{ k: string; v: string; ac?: boolean }> = [
    { k: 'compute', v: arch },
    { k: 'kernels verified', v: report ? `${passed}/${total}` : '…', ac: true },
    { k: 'precision', v: 'f32 · row-major' },
    { k: 'backend', v: 'WebGPU / WGSL' },
  ];
  return (
    <div className="overview">
      {cards.map((c) => (
        <div className="ov-card" key={c.k}>
          <div className="ov-k">{c.k}</div>
          <div className={`ov-v${c.ac ? ' ac' : ''}`}>{c.v}</div>
        </div>
      ))}
    </div>
  );
}

function fmtBytes(n: number): string {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(n % (1 << 30) ? 2 : 0)} GB`;
  if (n >= 1 << 20) return `${Math.round(n / (1 << 20))} MB`;
  return `${Math.round(n / 1024)} KB`;
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
      <div className={`summary ${report.allPassed ? 'ok' : 'bad'}`}>
        <span className="big">
          {report.allPassed ? `${passed} / ${total} NOMINAL` : `${total - passed} FAULT`}
        </span>
        <span className="sub">GPU kernels diffed against an f64 CPU oracle (rel. err &lt; 1e-3) · architecture invariants checked</span>
      </div>

      <div className="diag-wrap"><table className="diag">
        <thead>
          <tr>
            <th className="l">op</th>
            <th className="l">detail</th>
            <th>max abs err</th>
            <th>max rel err</th>
            <th>time</th>
            <th>throughput</th>
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>
          {GROUP_ORDER.map((group) => {
            const rows = report.results.filter((r) => r.group === group);
            if (rows.length === 0) return null;
            return (
              <Fragment key={group}>
                <tr className="grouprow">
                  <td colSpan={7}>
                    {GROUP_LABELS[group]} <span className="gcount">· {rows.length} checks</span>
                  </td>
                </tr>
                {rows.map((r, i) => (
                  <tr key={group + i}>
                    <td className="l strong">{r.name}</td>
                    <td className="l detail">{r.detail}</td>
                    <td>{r.maxAbsErr.toExponential(2)}</td>
                    <td>{r.maxRelErr.toExponential(2)}</td>
                    <td className="detail">{r.ms.toFixed(2)} ms</td>
                    <td className="detail">{r.throughput ?? '—'}</td>
                    <td className={`verdict ${r.pass ? 'pass' : 'fail'}`}>
                      {r.pass ? 'PASS' : 'FAIL'}
                    </td>
                  </tr>
                ))}
              </Fragment>
            );
          })}
        </tbody>
      </table></div>

      <div className="row" style={{ marginTop: 20 }}>
        <button className="btn" onClick={onRerun} disabled={running}>
          {running ? 'running…' : 're-run diagnostics'}
        </button>
        <span className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
          round-trip includes readback; throughput is indicative, not a benchmark.
        </span>
      </div>
    </>
  );
}

function TrainingCard({ t }: { t: TrainingResult }) {
  return (
    <section className="card">
      <h2>
        <span className="ch">CH0</span>Reference train · synthetic
      </h2>
      <div className={`banner ${t.pass ? 'pass' : 'fail'}`} style={{ marginBottom: 18 }}>
        <span className="big">{t.pass ? '✓ IT LEARNS' : '✗ NO CONVERGENCE'}</span>
        <span>
          loss {t.initialLoss.toFixed(2)} → {t.finalLoss.toFixed(3)} · accuracy{' '}
          {(t.accuracy * 100).toFixed(1)}%
        </span>
      </div>
      <div className="train-row">
        <div style={{ flex: 1, minWidth: 240 }}>
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

function Sparkline({ data, width = 600, height = 76 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const px = (i: number) => (i / (data.length - 1)) * width;
  const py = (v: number) => height - 4 - ((v - min) / range) * (height - 8);
  const line = data.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
  const area = `0,${height} ${line} ${width},${height}`;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="spark"
      style={{ width: '100%', height }}
      role="img"
      aria-label="training loss curve"
    >
      <polygon points={area} fill="rgba(236,72,153,0.20)" />
      <polyline
        points={line}
        fill="none"
        stroke="var(--pink)"
        strokeWidth={2.5}
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ProfilerCard() {
  const [rows, setRows] = useState<ProfileRow[] | null>(null);
  const [running, setRunning] = useState(false);
  const run = useCallback(async () => {
    setRunning(true);
    try {
      setRows(await runProfile());
    } finally {
      setRunning(false);
    }
  }, []);
  return (
    <section className="card" id="profile">
      <h2>
        <span className="ch">CH4</span>GPU profiler
        <span className="meta">amortized kernel throughput</span>
      </h2>
      {!rows && (
        <div className="row">
          <button className="btn" onClick={run} disabled={running}>
            {running ? 'profiling…' : 'run profiler'}
          </button>
          <span className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
            warms up, then times each kernel over many GPU dispatches (no readback)
          </span>
        </div>
      )}
      {rows && <ProfilerTable rows={rows} running={running} onRerun={run} />}
    </section>
  );
}

function ProfilerTable({
  rows,
  running,
  onRerun,
}: {
  rows: ProfileRow[];
  running: boolean;
  onRerun: () => void;
}) {
  const maxG = Math.max(1, ...rows.map((r) => r.gflops ?? 0));
  return (
    <>
      <div className="diag-wrap"><table className="diag">
        <thead>
          <tr>
            <th className="l">kernel</th>
            <th className="l">shape</th>
            <th>gpu time</th>
            <th className="l">throughput</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="l strong">{r.label}</td>
              <td className="l detail">{r.detail}</td>
              <td>{r.ms < 1 ? `${Math.round(r.ms * 1000)} µs` : `${r.ms.toFixed(2)} ms`}</td>
              <td className="l">
                {r.gflops != null ? (
                  <div className="perfwrap">
                    <div className="perfbar">
                      <div className="perfbar-fill" style={{ width: `${(r.gflops / maxG) * 100}%` }} />
                    </div>
                    <span>{r.gflops.toFixed(1)} GF/s</span>
                  </div>
                ) : (
                  <span className="detail">{Math.round(1000 / r.ms).toLocaleString()} steps/s</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table></div>
      <div className="row" style={{ marginTop: 18 }}>
        <button className="btn ghost" onClick={onRerun} disabled={running}>
          {running ? 'profiling…' : 're-run'}
        </button>
        <span className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
          amortized GPU time over many dispatches · no readback · real throughput
        </span>
      </div>
    </>
  );
}

function errMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
