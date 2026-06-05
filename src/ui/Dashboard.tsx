// The live training dashboard. A Web Worker runs continuous MNIST training on
// its own GPUDevice and streams metrics + a probe sample's activations back.
// High-frequency data is buffered in refs; a single requestAnimationFrame loop
// renders the loss chart and activation heatmaps at 60fps regardless of how fast
// the worker trains. The draw-a-digit demo round-trips a hand-drawn 28×28 image
// through the worker for live classification.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ActivationsMsg, EmbeddingMsg, InMsg, LandscapeMsg, OutMsg } from '../worker/protocol';

interface Info {
  adapter: string;
  train: number;
  test: number;
  pixels: number;
  hidden: number;
  lr: number;
}
interface Metrics {
  step: number;
  loss: number;
  trainAcc: number;
  stepsPerSec: number;
}

type TrainerPhase = 'idle' | 'loading' | 'ready' | 'error';

interface TrainerRefs {
  lossHist: { current: number[] };
  accHist: { current: number[] };
  activations: { current: ActivationsMsg | null };
  embedding: { current: EmbeddingMsg | null };
  landscape: { current: LandscapeMsg | null };
  metrics: { current: Metrics };
  testAcc: { current: number | null };
}

const HISTORY_CAP = 600;

function useTrainer() {
  const workerRef = useRef<Worker | null>(null);
  const [phase, setPhase] = useState<TrainerPhase>('idle');
  const [status, setStatus] = useState('');
  const [info, setInfo] = useState<Info | null>(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  const [inferProbs, setInferProbs] = useState<Float32Array | null>(null);
  const [landscapeBusy, setLandscapeBusy] = useState(false);

  const refs = useMemo<TrainerRefs>(
    () => ({
      lossHist: { current: [] },
      accHist: { current: [] },
      activations: { current: null },
      embedding: { current: null },
      landscape: { current: null },
      metrics: { current: { step: 0, loss: NaN, trainAcc: 0, stepsPerSec: 0 } },
      testAcc: { current: null },
    }),
    [],
  );

  const send = useCallback((msg: InMsg, transfer: Transferable[] = []) => {
    workerRef.current?.postMessage(msg, transfer);
  }, []);

  const begin = useCallback(() => {
    if (workerRef.current) return;
    setPhase('loading');
    setStatus('starting worker…');
    const worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<OutMsg>) => {
      const msg = e.data;
      switch (msg.type) {
        case 'status':
          setStatus(msg.message);
          break;
        case 'ready':
          setInfo({
            adapter: msg.adapter,
            train: msg.train,
            test: msg.test,
            pixels: msg.pixels,
            hidden: msg.hidden,
            lr: msg.lr,
          });
          setPhase('ready');
          setRunning(true);
          worker.postMessage({ type: 'start' } satisfies InMsg);
          break;
        case 'metrics':
          refs.metrics.current = msg;
          if (Number.isFinite(msg.loss)) {
            refs.lossHist.current.push(msg.loss);
            refs.accHist.current.push(msg.trainAcc);
            if (refs.lossHist.current.length > HISTORY_CAP) refs.lossHist.current.shift();
            if (refs.accHist.current.length > HISTORY_CAP) refs.accHist.current.shift();
          } else {
            // reset
            refs.lossHist.current = [];
            refs.accHist.current = [];
            refs.testAcc.current = null;
          }
          break;
        case 'testacc':
          refs.testAcc.current = msg.testAcc;
          break;
        case 'activations':
          refs.activations.current = msg;
          break;
        case 'embedding':
          refs.embedding.current = msg;
          break;
        case 'landscape':
          refs.landscape.current = msg;
          setLandscapeBusy(false);
          break;
        case 'probs':
          setInferProbs(msg.probs);
          break;
        case 'error':
          setError(msg.message);
          setPhase('error');
          break;
      }
    };
    worker.postMessage({ type: 'init' } satisfies InMsg);
  }, [refs]);

  const pause = useCallback(() => {
    send({ type: 'pause' });
    setRunning(false);
  }, [send]);
  const resume = useCallback(() => {
    send({ type: 'start' });
    setRunning(true);
  }, [send]);
  const reset = useCallback(
    (hidden: number, lr: number) => {
      send({ type: 'reset', hidden, lr });
      setInfo((prev) => (prev ? { ...prev, hidden, lr } : prev));
    },
    [send],
  );
  const infer = useCallback(
    (pixels: Float32Array) => send({ type: 'infer', pixels }, [pixels.buffer]),
    [send],
  );
  const clearInfer = useCallback(() => setInferProbs(null), []);
  const computeLandscape = useCallback(() => {
    setLandscapeBusy(true);
    send({ type: 'landscape' });
  }, [send]);

  useEffect(() => {
    return () => workerRef.current?.terminate();
  }, []);

  return {
    phase, status, info, error, running, inferProbs, landscapeBusy, refs,
    begin, pause, resume, reset, infer, clearInfer, computeLandscape,
  };
}

// ---- canvas drawing ----

function drawDigit(canvas: HTMLCanvasElement | null, input: Float32Array, side = 28): void {
  if (!canvas) return;
  if (canvas.width !== side) canvas.width = side;
  if (canvas.height !== side) canvas.height = side;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const v = Math.max(0, Math.min(1, input[i] ?? 0));
    const c = Math.round(v * 255);
    img.data[i * 4] = c;
    img.data[i * 4 + 1] = c;
    img.data[i * 4 + 2] = c;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Thermal colormap: cold near-black → blue → hot amber. The UI's accent palette,
// applied to the activations — so the heatmap and the chrome share one language.
const THERMAL: Array<[number, [number, number, number]]> = [
  [0.0, [21, 16, 31]],
  [0.3, [72, 40, 112]],
  [0.55, [139, 92, 246]],
  [0.78, [236, 72, 153]],
  [1.0, [249, 168, 212]],
];
function thermal(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < THERMAL.length; i++) {
    if (t <= THERMAL[i][0]) {
      const [a, ca] = THERMAL[i - 1];
      const [b, cb] = THERMAL[i];
      const u = (t - a) / (b - a);
      return [
        Math.round(ca[0] + (cb[0] - ca[0]) * u),
        Math.round(ca[1] + (cb[1] - ca[1]) * u),
        Math.round(ca[2] + (cb[2] - ca[2]) * u),
      ];
    }
  }
  return THERMAL[THERMAL.length - 1][1];
}

function drawHidden(canvas: HTMLCanvasElement | null, hidden: Float32Array): void {
  if (!canvas) return;
  const dim = hidden.length;
  const side = Math.ceil(Math.sqrt(dim));
  if (canvas.width !== side) canvas.width = side;
  if (canvas.height !== side) canvas.height = side;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let max = 1e-6;
  for (let i = 0; i < dim; i++) if (hidden[i] > max) max = hidden[i];
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const [r, g, b] = i < dim ? thermal(hidden[i] / max) : [8, 10, 14];
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawLoss(canvas: HTMLCanvasElement | null, loss: number[], acc: number[]): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // faint horizontal grid
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  for (let i = 1; i < 5; i++) {
    const y = Math.round((H / 5) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  const n = loss.length;
  if (n < 2) return;
  let maxLoss = 1e-6;
  for (let i = 0; i < n; i++) if (loss[i] > maxLoss) maxLoss = loss[i];
  const pad = 4 * dpr;
  const yOf = (v: number, max: number) => H - pad - (v / max) * (H - 2 * pad);
  const xOf = (i: number) => (i / (n - 1)) * W;

  // accuracy — purple
  ctx.lineWidth = 1.8 * dpr;
  ctx.strokeStyle = '#c084fc';
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i);
    const y = yOf(acc[i], 1);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // loss — violet line with a gradient area fill
  ctx.beginPath();
  ctx.moveTo(0, yOf(loss[0], maxLoss));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(loss[i], maxLoss));
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(236,72,153,0.38)');
  grad.addColorStop(1, 'rgba(236,72,153,0)');
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.lineWidth = 2.8 * dpr;
  ctx.strokeStyle = '#f472b6';
  ctx.shadowBlur = 0;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i);
    const y = yOf(loss[i], maxLoss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(W - 1, yOf(loss[n - 1], maxLoss), 3 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = '#f472b6';
  ctx.fill();
  ctx.shadowBlur = 0;
}

// 10-class categorical palette for the embedding scatter
const CLASS_COLORS = [
  '#a78bfa', '#ec4899', '#38d6ff', '#f59e0b', '#4ade80',
  '#fb923c', '#818cf8', '#f43f5e', '#facc15', '#2dd4bf',
];

function drawScatter(canvas: HTMLCanvasElement | null, emb: EmbeddingMsg | null): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!emb) return;
  const { coords, labels } = emb;
  const n = labels.length;
  const pad = 16 * dpr;
  const r = 2.6 * dpr;
  ctx.globalAlpha = 0.9;
  for (let i = 0; i < n; i++) {
    const x = pad + ((coords[i * 2] + 1) / 2) * (W - 2 * pad);
    const y = pad + ((coords[i * 2 + 1] + 1) / 2) * (H - 2 * pad);
    ctx.fillStyle = CLASS_COLORS[labels[i]] ?? '#fff';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawLandscape(canvas: HTMLCanvasElement | null, land: LandscapeMsg | null): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!land) return;

  const { grid, size: G } = land;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] < min) min = grid[i];
    if (grid[i] > max) max = grid[i];
  }
  const span = max - min || 1;
  const norm = (h: number) => (h - min) / span;

  const cx = W / 2;
  const cyTop = H * 0.4;
  const sx = (W * 0.42) / G;
  const sy = (H * 0.27) / G;
  const amp = H * 0.33;
  const px = (i: number, j: number) => cx + (i - j) * sx;
  const py = (i: number, j: number, h: number) => cyTop + (i + j) * sy - norm(h) * amp;

  const lo = [124, 92, 246]; // low loss → purple
  const hi = [236, 72, 153]; // high loss → pink
  const color = (t: number) =>
    `rgb(${Math.round(lo[0] + (hi[0] - lo[0]) * t)},${Math.round(lo[1] + (hi[1] - lo[1]) * t)},${Math.round(lo[2] + (hi[2] - lo[2]) * t)})`;

  const quads: Array<[number, number]> = [];
  for (let i = 0; i < G - 1; i++) for (let j = 0; j < G - 1; j++) quads.push([i, j]);
  quads.sort((a, b) => a[0] + a[1] - (b[0] + b[1])); // far → near

  ctx.lineWidth = 1;
  for (const [i, j] of quads) {
    const h00 = grid[i * G + j];
    const h01 = grid[i * G + j + 1];
    const h10 = grid[(i + 1) * G + j];
    const h11 = grid[(i + 1) * G + j + 1];
    ctx.beginPath();
    ctx.moveTo(px(i, j), py(i, j, h00));
    ctx.lineTo(px(i + 1, j), py(i + 1, j, h10));
    ctx.lineTo(px(i + 1, j + 1), py(i + 1, j + 1, h11));
    ctx.lineTo(px(i, j + 1), py(i, j + 1, h01));
    ctx.closePath();
    ctx.fillStyle = color(norm((h00 + h01 + h10 + h11) / 4));
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.28)';
    ctx.stroke();
  }

  // current weights (grid center, α=β=0)
  const m = (G - 1) / 2;
  const hc = grid[Math.round(m) * G + Math.round(m)];
  ctx.beginPath();
  ctx.arc(px(m, m), py(m, m, hc), 4 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.strokeStyle = '#15101f';
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ---- output bars ----

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

function ProbBars({ probs, pred }: { probs: Float32Array | null; pred: number }) {
  return (
    <div className="bars">
      {DIGITS.map((d) => {
        const p = probs ? probs[d] : 0;
        return (
          <div key={d} className="bar-col" title={`${(p * 100).toFixed(1)}%`}>
            <div className="bar-track">
              <div
                className={`bar-fill${d === pred ? ' top' : ''}`}
                style={{ height: `${Math.max(2, p * 100)}%` }}
              />
            </div>
            <span className={d === pred ? 'top' : ''}>{d}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---- main dashboard ----

interface Readout extends Metrics {
  testAcc: number | null;
  label: number;
  pred: number;
  probs: Float32Array | null;
}

export default function Dashboard() {
  const t = useTrainer();
  const lossRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLCanvasElement>(null);
  const hiddenRef = useRef<HTMLCanvasElement>(null);
  const embedRef = useRef<HTMLCanvasElement>(null);
  const landscapeRef = useRef<HTMLCanvasElement>(null);
  const [readout, setReadout] = useState<Readout>({
    step: 0,
    loss: NaN,
    trainAcc: 0,
    stepsPerSec: 0,
    testAcc: null,
    label: 0,
    pred: 0,
    probs: null,
  });

  // single rAF render loop
  const refs = t.refs;
  useEffect(() => {
    if (t.phase !== 'ready') return;
    let raf = 0;
    let frame = 0;
    const loop = () => {
      const act = refs.activations.current;
      if (act) {
        drawDigit(inputRef.current, act.input);
        drawHidden(hiddenRef.current, act.hidden);
      }
      drawLoss(lossRef.current, refs.lossHist.current, refs.accHist.current);
      drawScatter(embedRef.current, refs.embedding.current);
      drawLandscape(landscapeRef.current, refs.landscape.current);
      if (++frame % 4 === 0) {
        const m = refs.metrics.current;
        setReadout({
          ...m,
          testAcc: refs.testAcc.current,
          label: act?.label ?? 0,
          pred: act?.pred ?? 0,
          probs: act?.probs ?? null,
        });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [t.phase, refs]);

  if (t.phase === 'idle') {
    return (
      <section className="card">
        <h2>
        <span className="ch">CH1</span>Live training · MNIST
      </h2>
        <div className="row">
          <button className="btn" onClick={t.begin}>
            ▶ start live training
          </button>
          <span className="muted" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
            spawns a worker, loads MNIST (~11 MB), and trains continuously off the main thread
          </span>
        </div>
      </section>
    );
  }

  if (t.phase === 'error') {
    return (
      <section className="card">
        <h2>
          <span className="ch">CH1</span>Live training
        </h2>
        <p className="banner fail">
          <code className="inline">{t.error}</code>
        </p>
      </section>
    );
  }

  if (t.phase === 'loading') {
    return (
      <section className="card">
        <h2>
        <span className="ch">CH1</span>Live training · MNIST
      </h2>
        <p className="banner info">{t.status}</p>
      </section>
    );
  }

  // ready
  return (
    <section className="card" id="live">
      <h2>
        <span className="ch">CH1</span>Live training · MNIST
        <span className="meta">784 → {t.info?.hidden} → 10 · Adam · lr 5.0e-3</span>
      </h2>

      <div className="dash-top">
        <div className="metrics-grid">
          <Stat label="step" value={readout.step.toLocaleString()} />
          <Stat
            label="loss"
            value={Number.isFinite(readout.loss) ? readout.loss.toFixed(3) : '—'}
            accent="amber"
          />
          <Stat label="train acc" value={`${(readout.trainAcc * 100).toFixed(1)}%`} />
          <Stat
            label="test acc"
            value={readout.testAcc == null ? '…' : `${(readout.testAcc * 100).toFixed(1)}%`}
            accent="green"
          />
          <Stat
            label="steps/s"
            value={Math.round(readout.stepsPerSec).toLocaleString()}
            accent="blue"
          />
        </div>
        <div className="dash-controls">
          {t.running ? (
            <button className="btn ghost" onClick={t.pause}>
              ❚❚ pause
            </button>
          ) : (
            <button className="btn" onClick={t.resume}>
              ▶ resume
            </button>
          )}
          <button className="btn ghost" onClick={() => t.reset(t.info?.hidden ?? 64, t.info?.lr ?? 0.005)}>
            ↺ reset
          </button>
          <HiddenSelect
            value={t.info?.hidden ?? 64}
            onChange={(h) => t.reset(h, t.info?.lr ?? 0.005)}
          />
        </div>
      </div>

      <div className="scope-grid">
        <div>
          <canvas ref={lossRef} width={800} height={150} className="loss-canvas" />
          <div className="spark-cap">
            <span style={{ color: 'var(--pink)' }}>━</span> loss ·{' '}
            <span style={{ color: 'var(--purple)' }}>━</span> train accuracy
          </div>
        </div>
        <div className="probe">
          <div className="probe-cell">
            <canvas ref={inputRef} className="px cold" width={28} height={28} />
            <span>input · "{readout.label}"</span>
          </div>
          <div className="probe-cell">
            <canvas ref={hiddenRef} className="px hot" width={8} height={8} />
            <span>hidden [{t.info?.hidden}]</span>
          </div>
          <div className="probe-cell out">
            <ProbBars probs={readout.probs} pred={readout.pred} />
            <span>output → {readout.pred}</span>
          </div>
        </div>
      </div>

      <div className="group-label" style={{ marginTop: 24 }}>
        ▌ embedding · hidden activations → PCA · watch the classes separate
      </div>
      <div className="embed-row">
        <canvas ref={embedRef} className="embed-canvas" width={900} height={300} />
        <div className="embed-legend">
          {CLASS_COLORS.map((c, d) => (
            <span key={d}>
              <i style={{ background: c }} />
              {d}
            </span>
          ))}
        </div>
      </div>

      <div className="group-label" style={{ marginTop: 24 }}>
        ▌ loss landscape · loss around the weights, two random directions
      </div>
      <div className="land-row">
        <canvas ref={landscapeRef} className="land-canvas" width={560} height={300} />
        <div className="land-side">
          <button className="btn ghost" onClick={t.computeLandscape} disabled={t.landscapeBusy}>
            {t.landscapeBusy ? 'sampling…' : 'compute surface'}
          </button>
          <p>
            samples the loss on a 21×21 grid around the current weights along two filter-normalized
            directions. purple = low, pink = high. white dot = the current weights.
          </p>
        </div>
      </div>

      <DrawDemo
        pixels={t.info?.pixels ?? 784}
        onInfer={t.infer}
        onClear={t.clearInfer}
        probs={t.inferProbs}
      />
    </section>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: 'amber' | 'blue' | 'green';
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${accent ? ' ' + accent : ''}`}>{value}</div>
    </div>
  );
}

function HiddenSelect({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <label className="select">
      hidden
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {[32, 64, 128, 256].map((h) => (
          <option key={h} value={h}>
            {h}
          </option>
        ))}
      </select>
    </label>
  );
}

// ---- draw-a-digit ----

function DrawDemo({
  pixels,
  onInfer,
  onClear,
  probs,
}: {
  pixels: number;
  onInfer: (p: Float32Array) => void;
  onClear: () => void;
  probs: Float32Array | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const lastInfer = useRef(0);
  const side = Math.round(Math.sqrt(pixels)) || 28;

  const clear = useCallback(() => {
    const c = canvasRef.current;
    const ctx = c?.getContext('2d');
    if (c && ctx) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, c.width, c.height);
    }
    onClear();
  }, [onClear]);

  useEffect(() => {
    clear();
  }, [clear]);

  // Match MNIST preprocessing: crop to the ink's bounding box, scale the longest
  // side to 20px, then center by center-of-mass in a 28×28 field. Without this,
  // a raw drawing is wildly off the training distribution and misclassifies.
  const extract = useCallback((): Float32Array => {
    const c = canvasRef.current!;
    const sctx = c.getContext('2d')!;
    const W = c.width;
    const H = c.height;
    const srcData = sctx.getImageData(0, 0, W, H).data;
    const out = new Float32Array(side * side);

    let minX = W, minY = H, maxX = -1, maxY = -1;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (srcData[(y * W + x) * 4] > 24) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return out; // nothing drawn

    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    const dw = Math.max(1, Math.round(bw * scale));
    const dh = Math.max(1, Math.round(bh * scale));

    const tmp = document.createElement('canvas');
    tmp.width = side;
    tmp.height = side;
    const tctx = tmp.getContext('2d')!;
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(
      c,
      minX, minY, bw, bh,
      Math.round((side - dw) / 2), Math.round((side - dh) / 2), dw, dh,
    );

    // shift so center-of-mass lands at the center (MNIST centers by CoM)
    let px = tctx.getImageData(0, 0, side, side).data;
    let sum = 0, cx = 0, cy = 0;
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const v = px[(y * side + x) * 4];
        sum += v;
        cx += x * v;
        cy += y * v;
      }
    }
    if (sum > 0) {
      const sx = Math.round(side / 2 - cx / sum);
      const sy = Math.round(side / 2 - cy / sum);
      if (sx !== 0 || sy !== 0) {
        const shifted = document.createElement('canvas');
        shifted.width = side;
        shifted.height = side;
        const shctx = shifted.getContext('2d')!;
        shctx.drawImage(tmp, sx, sy);
        px = shctx.getImageData(0, 0, side, side).data;
      }
    }

    for (let i = 0; i < out.length; i++) out[i] = px[i * 4] / 255;
    return out;
  }, [side]);

  const pos = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * canvasRef.current!.width,
      y: ((e.clientY - r.top) / r.height) * canvasRef.current!.height,
    };
  };

  const down = (e: React.PointerEvent) => {
    drawing.current = true;
    last.current = pos(e);
    try {
      (e.target as Element).setPointerCapture(e.pointerId);
    } catch {
      /* pointer may not be capturable (synthetic events) */
    }
  };
  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const c = canvasRef.current!;
    const ctx = c.getContext('2d')!;
    const p = pos(e);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(last.current!.x, last.current!.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
    const now = performance.now();
    if (now - lastInfer.current > 100) {
      lastInfer.current = now;
      onInfer(extract());
    }
  };
  const up = () => {
    if (!drawing.current) return;
    drawing.current = false;
    onInfer(extract());
  };

  let pred = -1;
  if (probs) {
    let best = -1;
    for (let i = 0; i < probs.length; i++)
      if (probs[i] > best) {
        best = probs[i];
        pred = i;
      }
  }
  const conf = probs && pred >= 0 ? probs[pred] : 0;

  return (
    <div className="draw" id="draw" style={{ scrollMarginTop: 90 }}>
      <div className="group-label" style={{ marginTop: 22 }}>
        ▌ draw a digit → live GPU inference
      </div>
      <div className="draw-row">
        <div className="draw-pad">
          <canvas
            ref={canvasRef}
            width={252}
            height={252}
            className="draw-canvas"
            onPointerDown={down}
            onPointerMove={move}
            onPointerUp={up}
            onPointerLeave={up}
          />
          <button className="btn ghost draw-clear" onClick={clear}>
            clear
          </button>
        </div>
        <div className="draw-result">
          <div className="draw-result-top">
            <div className="prediction">{pred >= 0 ? pred : '–'}</div>
            <div className="draw-conf">
              <div className="draw-hint-v">
                {pred >= 0 ? `${Math.round(conf * 100)}%` : '—'}
                <span>confidence</span>
              </div>
              <p>
                every stroke runs a full forward pass on the GPU worker — the bars below are the
                live softmax over all ten classes.
              </p>
            </div>
          </div>
          <ProbBars probs={probs} pred={pred} />
        </div>
      </div>
    </div>
  );
}
