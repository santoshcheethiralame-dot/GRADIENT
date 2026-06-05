import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { getGpuContext, WebGpuUnsupportedError } from './gpu/device';
import {
  runSelfTest,
  type SelfTestReport,
  type CheckGroup,
  type TrainingResult,
} from './gpu/selftest';
import Dashboard from './ui/Dashboard';

interface DeviceLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
}

type Phase =
  | { kind: 'init' }
  | { kind: 'ready'; label: string; info: GPUAdapterInfo; limits: DeviceLimits }
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
        → SGD/Adam → MNIST → a Worker-driven live dashboard with activation heatmaps and draw-a-digit inference.
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
        <span className="sub">all kernels verified against an f64 CPU oracle · rel. err &lt; 1e-3</span>
      </div>

      <table className="diag">
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
      </table>

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

function errMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
