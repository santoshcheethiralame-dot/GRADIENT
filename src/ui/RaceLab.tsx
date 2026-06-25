import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt, NANO_CORPUS } from '../nn/nanogpt';
import { GpuNanoGpt } from '../nn/nanogpt-gpu';
import { getGpuContext } from '../gpu/device';
import { mulberry32 } from '../data/synthetic';
import type { RaceOut } from '../worker/race-protocol';

const DUR_MS = 10000;
const REPORT_MS = 120;
const T = 64;
const LR = 0.0025;
const SEED = 5;
const D_EMBED = 128;
const D_FF = 256;

const CPU_COLOR = '#ffd24a';
const GPU_COLOR = '#3aa0ff';

type Phase = 'idle' | 'racing' | 'done';
type Lane = { step: number; loss: number; ms: number };
type Pt = { t: number; loss: number };

const ZERO: Lane = { step: 0, loss: NaN, ms: 0 };

function sps(l: Lane): number {
  return l.ms > 0 ? l.step / (l.ms / 1000) : 0;
}

function drawChart(canvas: HTMLCanvasElement | null, cpu: Pt[], gpu: Pt[]): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0d16';
  ctx.fillRect(0, 0, w, h);

  const all = cpu.concat(gpu);
  if (all.length < 2) return;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of all) {
    if (Number.isNaN(p.loss)) continue;
    if (p.loss < lo) lo = p.loss;
    if (p.loss > hi) hi = p.loss;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  const padX = 6 * dpr;
  const padY = 6 * dpr;
  const x = (t: number): number => padX + (t / DUR_MS) * (w - 2 * padX);
  const y = (v: number): number => padY + (1 - (v - lo) / (hi - lo || 1)) * (h - 2 * padY);

  const line = (pts: Pt[], color: string): void => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (const p of pts) {
      if (Number.isNaN(p.loss)) continue;
      const px = x(p.t);
      const py = y(p.loss);
      if (started) ctx.lineTo(px, py);
      else {
        ctx.moveTo(px, py);
        started = true;
      }
    }
    ctx.stroke();
  };
  line(cpu, CPU_COLOR);
  line(gpu, GPU_COLOR);
}

export function RaceLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [cpu, setCpu] = useState<Lane>(ZERO);
  const [gpu, setGpu] = useState<Lane>(ZERO);
  const cpuHist = useRef<Pt[]>([]);
  const gpuHist = useRef<Pt[]>([]);
  const chart = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const gpuRef = useRef<GpuNanoGpt | null>(null);
  const running = useRef(false);
  const gpuDone = useRef(false);
  const cpuDone = useRef(false);

  const teardown = useCallback(() => {
    running.current = false;
    const w = workerRef.current;
    if (w) {
      w.postMessage({ type: 'stop' });
      w.terminate();
      workerRef.current = null;
    }
  }, []);

  useEffect(() => () => teardown(), [teardown]);

  const stop = useCallback(() => {
    teardown();
    setPhase('done');
  }, [teardown]);

  const finish = useCallback(() => {
    if (gpuDone.current && cpuDone.current) {
      drawChart(chart.current, cpuHist.current, gpuHist.current);
      setPhase('done');
    }
  }, []);

  const race = useCallback(() => {
    teardown();
    running.current = true;
    gpuDone.current = false;
    cpuDone.current = false;
    cpuHist.current = [];
    gpuHist.current = [];
    setCpu(ZERO);
    setGpu(ZERO);
    setPhase('racing');
    drawChart(chart.current, [], []);
    void (async () => {
      const { device } = await getGpuContext();
      const tok = new CharTokenizer(NANO_CORPUS);
      const reps = Math.max(2, Math.ceil((T * 4) / NANO_CORPUS.length));
      const data = tok.encode(NANO_CORPUS.repeat(reps));
      const cfg = { vocab: tok.vocab, dEmbed: D_EMBED, dFF: D_FF, blockSize: T };
      const seedModel = new NanoGpt(cfg, SEED);
      const gpuModel = new GpuNanoGpt(device, seedModel);
      gpuRef.current = gpuModel;

      const worker = new Worker(new URL('../worker/race.worker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;
      worker.onmessage = (e: MessageEvent<RaceOut>): void => {
        const m = e.data;
        cpuHist.current.push({ t: m.elapsedMs, loss: m.loss });
        setCpu({ step: m.step, loss: m.loss, ms: m.elapsedMs });
        drawChart(chart.current, cpuHist.current, gpuHist.current);
        if (m.type === 'done') {
          cpuDone.current = true;
          finish();
        }
      };
      worker.postMessage({
        type: 'start',
        cfg,
        seed: SEED,
        data,
        T,
        lr: LR,
        durationMs: DUR_MS,
        reportMs: REPORT_MS,
      });

      const grng = mulberry32(17);
      const t0 = performance.now();
      let gstep = 0;
      let glast = NaN;
      let lastUi = 0;
      const loop = async (): Promise<void> => {
        if (!running.current) {
          gpuModel.destroy();
          gpuRef.current = null;
          return;
        }
        const elapsed = performance.now() - t0;
        if (elapsed >= DUR_MS) {
          setGpu({ step: gstep, loss: glast, ms: elapsed });
          drawChart(chart.current, cpuHist.current, gpuHist.current);
          gpuModel.destroy();
          gpuRef.current = null;
          gpuDone.current = true;
          finish();
          return;
        }
        const start = Math.floor(grng() * (data.length - T - 1));
        const loss = await gpuModel.step(
          data.slice(start, start + T),
          data.slice(start + 1, start + T + 1),
          LR,
        );
        gstep++;
        glast = loss;
        const now = performance.now() - t0;
        gpuHist.current.push({ t: now, loss });
        if (now - lastUi >= REPORT_MS) {
          setGpu({ step: gstep, loss, ms: now });
          drawChart(chart.current, cpuHist.current, gpuHist.current);
          lastUi = now;
        }
        setTimeout(loop, 0);
      };
      setTimeout(loop, 0);
    })();
  }, [finish, teardown]);

  const ratio = sps(cpu) > 0 ? sps(gpu) / sps(cpu) : 0;

  return (
    <section className="card nlab" id="race">
      <h2>
        <span className="ch">CH7</span>same model, two processors — CPU vs GPU, live
        <span className="meta">identical from-scratch GPT · who trains faster?</span>
      </h2>

      <div className="race-body">
      <p className="nlab-reqs">
        The exact same transformer, same weights, same data — trained two ways at once: one on a
        single CPU thread (in a Web Worker), one on your GPU in WGSL. Both run forward, backprop and
        Adam every step. Watch the GPU pull away.
      </p>

      <div className="race-grid">
        <div className={phase === 'racing' ? 'race-lane gpu live' : 'race-lane gpu'}>
          <span className="race-tag" style={{ color: GPU_COLOR }}>
            ● GPU · WGSL
          </span>
          <b className="race-steps">{gpu.step.toLocaleString()}</b>
          <span className="race-sub">steps</span>
          <div className="race-metrics">
            <span>{Math.round(sps(gpu)).toLocaleString()} steps/s</span>
            <span>loss {Number.isNaN(gpu.loss) ? '—' : gpu.loss.toFixed(3)}</span>
          </div>
        </div>
        <div className={phase === 'racing' ? 'race-lane cpu live' : 'race-lane cpu'}>
          <span className="race-tag" style={{ color: CPU_COLOR }}>
            ● CPU · 1 thread
          </span>
          <b className="race-steps">{cpu.step.toLocaleString()}</b>
          <span className="race-sub">steps</span>
          <div className="race-metrics">
            <span>{Math.round(sps(cpu)).toLocaleString()} steps/s</span>
            <span>loss {Number.isNaN(cpu.loss) ? '—' : cpu.loss.toFixed(3)}</span>
          </div>
        </div>
      </div>

      <div className="race-headline">
        {ratio > 0 ? (
          <>
            <b>{ratio.toFixed(1)}×</b>
            <span>{phase === 'done' ? 'GPU was faster' : 'GPU faster, live'}</span>
          </>
        ) : (
          <span className="race-hint">press race — both train the same GPT for {DUR_MS / 1000}s</span>
        )}
      </div>

      <div className="race-chartwrap">
        <div className="race-legend">
          <span style={{ color: GPU_COLOR }}>● GPU</span>
          <span style={{ color: CPU_COLOR }}>● CPU</span>
          <span className="race-axis">loss vs. wall-clock ({DUR_MS / 1000}s)</span>
        </div>
        <canvas ref={chart} className="race-chart" />
      </div>

      <div className="nlab-controls">
        {phase === 'racing' ? (
          <button className="btn" onClick={stop}>
            stop
          </button>
        ) : (
          <button className="btn primary" onClick={race}>
            {phase === 'done' ? 'race again' : 'race'}
          </button>
        )}
      </div>
      </div>
    </section>
  );
}
