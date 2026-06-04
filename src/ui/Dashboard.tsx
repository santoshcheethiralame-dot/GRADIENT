// The live training dashboard. A Web Worker runs continuous MNIST training on
// its own GPUDevice and streams metrics + a probe sample's activations back.
// High-frequency data is buffered in refs; a single requestAnimationFrame loop
// renders the loss chart and activation heatmaps at 60fps regardless of how fast
// the worker trains. The draw-a-digit demo round-trips a hand-drawn 28×28 image
// through the worker for live classification.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { scaleSequential, interpolateViridis, rgb } from 'd3';
import type { ActivationsMsg, InMsg, OutMsg } from '../worker/protocol';

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

  const refs = useMemo<TrainerRefs>(
    () => ({
      lossHist: { current: [] },
      accHist: { current: [] },
      activations: { current: null },
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

  useEffect(() => {
    return () => workerRef.current?.terminate();
  }, []);

  return { phase, status, info, error, running, inferProbs, refs, begin, pause, resume, reset, infer };
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
  const scale = scaleSequential(interpolateViridis).domain([0, max]);
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    let r = 8;
    let g = 8;
    let b = 8;
    if (i < dim) {
      const col = rgb(scale(hidden[i]));
      r = col.r;
      g = col.g;
      b = col.b;
    }
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
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = loss.length;
  if (n < 2) return;
  let maxLoss = 1e-6;
  for (let i = 0; i < n; i++) if (loss[i] > maxLoss) maxLoss = loss[i];

  // accuracy line (0..1), amber
  ctx.strokeStyle = '#ffc44d';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H - 2 - acc[i] * (H - 4);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // loss line, lime
  ctx.strokeStyle = '#c8f140';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H - 2 - (loss[i] / maxLoss) * (H - 4);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
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
        <h2>Live training · MNIST on a Web Worker</h2>
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
        <h2>Live training</h2>
        <p className="banner fail">
          <code className="inline">{t.error}</code>
        </p>
      </section>
    );
  }

  if (t.phase === 'loading') {
    return (
      <section className="card">
        <h2>Live training · MNIST on a Web Worker</h2>
        <p className="banner info">{t.status}</p>
      </section>
    );
  }

  // ready
  return (
    <section className="card">
      <h2>Live training · MNIST on a Web Worker</h2>

      <div className="dash-top">
        <div className="metrics-grid">
          <Stat label="step" value={readout.step.toLocaleString()} />
          <Stat label="loss" value={Number.isFinite(readout.loss) ? readout.loss.toFixed(3) : '—'} />
          <Stat label="train acc" value={`${(readout.trainAcc * 100).toFixed(1)}%`} />
          <Stat
            label="test acc"
            value={readout.testAcc == null ? '…' : `${(readout.testAcc * 100).toFixed(1)}%`}
            accent
          />
          <Stat label="steps/s" value={Math.round(readout.stepsPerSec).toLocaleString()} />
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

      <div className="train-row" style={{ marginTop: 16 }}>
        <div>
          <canvas ref={lossRef} width={420} height={120} className="loss-canvas" />
          <div className="muted spark-cap">
            <span style={{ color: 'var(--lime)' }}>━</span> loss ·{' '}
            <span style={{ color: 'var(--amber)' }}>━</span> train accuracy
          </div>
        </div>
        <div className="probe">
          <div className="probe-cell">
            <canvas ref={inputRef} className="px" width={28} height={28} />
            <span>input · "{readout.label}"</span>
          </div>
          <div className="probe-cell">
            <canvas ref={hiddenRef} className="px" width={8} height={8} />
            <span>hidden ({t.info?.hidden})</span>
          </div>
          <div className="probe-cell out">
            <ProbBars probs={readout.probs} pred={readout.pred} />
            <span>output → {readout.pred}</span>
          </div>
        </div>
      </div>

      <DrawDemo pixels={t.info?.pixels ?? 784} onInfer={t.infer} probs={t.inferProbs} />
    </section>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${accent ? ' accent' : ''}`}>{value}</div>
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
  probs,
}: {
  pixels: number;
  onInfer: (p: Float32Array) => void;
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
    if (!c || !ctx) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, c.width, c.height);
  }, []);

  useEffect(() => {
    clear();
  }, [clear]);

  const extract = useCallback((): Float32Array => {
    const c = canvasRef.current!;
    const tmp = document.createElement('canvas');
    tmp.width = side;
    tmp.height = side;
    const tctx = tmp.getContext('2d')!;
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(c, 0, 0, side, side);
    const d = tctx.getImageData(0, 0, side, side).data;
    const out = new Float32Array(side * side);
    for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255; // white ink on black
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

  return (
    <div className="draw">
      <div className="group-label" style={{ marginTop: 22 }}>
        draw a digit → live GPU inference
      </div>
      <div className="draw-row">
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
        <div className="draw-out">
          <div className="prediction">{pred >= 0 ? pred : '–'}</div>
          <ProbBars probs={probs} pred={pred} />
        </div>
        <button className="btn ghost" onClick={clear}>
          clear
        </button>
      </div>
    </div>
  );
}
