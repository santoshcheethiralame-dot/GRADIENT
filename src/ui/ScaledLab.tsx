import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt } from '../nn/nanogpt';
import { GpuNanoGpt } from '../nn/nanogpt-gpu';
import { getGpuContext } from '../gpu/device';
import { mulberry32 } from '../data/synthetic';
import { ember } from './ember';

const CORPUS = `to be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles
and by opposing end them. to die—to sleep,
no more; and by a sleep to say we end
the heart-ache and the thousand natural shocks
that flesh is heir to: 'tis a consummation
devoutly to be wish'd. to die, to sleep;
to sleep, perchance to dream—ay, there's the rub:
for in that sleep of death what dreams may come,
when we have shuffled off this mortal coil,
must give us pause.

shall i compare thee to a summer's day?
thou art more lovely and more temperate:
rough winds do shake the darling buds of may,
and summer's lease hath all too short a date;
sometime too hot the eye of heaven shines,
and often is his gold complexion dimm'd. `;

const T = 64;
const STEPS = 900;
const CHUNK = 4;
const LR = 0.0025;
const GEN = 240;
const D_EMBED = 128;
const D_FF = 256;
const N_HEADS = 4;

type Phase = 'idle' | 'training' | 'done';

function drawAttn(canvas: HTMLCanvasElement | null, data: Float32Array | null, n: number): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const H = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#090d18';
  ctx.fillRect(0, 0, W, H);
  if (!data) return;
  const i = n - 1;
  let rowMax = 1e-6;
  for (let j = 0; j <= i; j++) if (data[i * n + j] > rowMax) rowMax = data[i * n + j];
  const rows = i + 1;
  const rowH = H / rows;
  const x0 = 4 * dpr;
  const span = W - 8 * dpr;
  for (let j = 0; j <= i; j++) {
    const rel = data[i * n + j] / rowMax;
    const [r, g, b] = ember(rel);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x0, j * rowH + rowH * 0.14, rel * span, Math.max(1, rowH * 0.72));
  }
}

export function ScaledLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(NaN);
  const [toks, setToks] = useState(0);
  const [sample, setSample] = useState('');
  const attnCanvas = useRef<HTMLCanvasElement>(null);
  const running = useRef(false);
  const gpuRef = useRef<GpuNanoGpt | null>(null);

  const stop = useCallback(() => {
    running.current = false;
    setPhase('done');
  }, []);

  const reset = useCallback(() => {
    running.current = false;
    drawAttn(attnCanvas.current, null, T);
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setToks(0);
    setSample('');
  }, []);

  const train = useCallback(() => {
    running.current = false;
    setPhase('training');
    setStep(0);
    setLoss(NaN);
    setToks(0);
    setSample('');
    void (async () => {
      const { device } = await getGpuContext();
      const tok = new CharTokenizer(CORPUS);
      const data = tok.encode(CORPUS);
      const cfg = { vocab: tok.vocab, dEmbed: D_EMBED, dFF: D_FF, blockSize: T, nHeads: N_HEADS };
      const cpu = new NanoGpt(cfg, 7);
      const gpu = new GpuNanoGpt(device, cpu);
      gpuRef.current = gpu;
      const rng = mulberry32(11);
      const probe = data.slice(0, T);
      const seed = CORPUS.slice(0, 12);
      const t0 = performance.now();
      running.current = true;
      let s = 0;
      const loop = async (): Promise<void> => {
        if (!running.current) {
          gpu.destroy();
          gpuRef.current = null;
          return;
        }
        let l = NaN;
        for (let c = 0; c < CHUNK && s < STEPS; c++) {
          s++;
          const start = Math.floor(rng() * (data.length - T - 1));
          l = await gpu.step(data.slice(start, start + T), data.slice(start + 1, start + T + 1), LR);
        }
        drawAttn(attnCanvas.current, await gpu.attention(probe), T);
        setStep(s);
        setLoss(l);
        setToks(Math.round((s * T) / ((performance.now() - t0) / 1000)));
        if (s < STEPS && running.current) {
          setTimeout(loop, 0);
        } else {
          await gpu.syncTo(cpu);
          setSample(tok.decode(cpu.generate(tok.encode(seed), GEN, 0.5, mulberry32(0))));
          gpu.destroy();
          gpuRef.current = null;
          if (s >= STEPS) setPhase('done');
        }
      };
      setTimeout(loop, 0);
    })();
  }, []);

  useEffect(
    () => () => {
      running.current = false;
    },
    [],
  );

  const pct = Math.round((step / STEPS) * 100);

  return (
    <section className="card nlab" id="scaled">
      <h2>
        <span className="ch">CH9</span>scaled transformer — multi-head, on the GPU
        <span className="meta">{D_EMBED}-dim · {N_HEADS} heads · {T}-token context</span>
      </h2>

      <div className="nlab-grid">
        <div className="nlab-left">
          <p className="nlab-reqs">
            The same from-scratch stack, scaled up: a <b>{N_HEADS}-head</b> attention block at{' '}
            {D_EMBED} dimensions over a {T}-character context, trained <b>entirely on your GPU</b> on a
            page of Shakespeare. Forward, backprop and Adam all in WGSL. The attention matrix is head 1
            of {N_HEADS}.
          </p>
          <div className="nlab-controls">
            {phase === 'training' ? (
              <button className="btn" onClick={stop}>
                stop
              </button>
            ) : (
              <button className="btn primary" onClick={train}>
                {phase === 'done' ? 'train again' : 'train on GPU'}
              </button>
            )}
            {phase !== 'idle' && (
              <button className="btn ghost" onClick={reset}>
                reset
              </button>
            )}
          </div>
          <div className="nlab-stats">
            <div>
              <b>{step}</b>
              <span>/ {STEPS} steps</span>
            </div>
            <div>
              <b>{Number.isNaN(loss) ? '—' : loss.toFixed(3)}</b>
              <span>loss (nats/char)</span>
            </div>
            <div>
              <b>{toks ? toks.toLocaleString() : '—'}</b>
              <span>tokens / sec</span>
            </div>
          </div>
          <div className="attn-wrap">
            <span className="attn-label">
              attention · head 1 · last token's reach over {T} positions
              <span className="ember-key" />
            </span>
            <canvas ref={attnCanvas} className="attn-canvas" />
          </div>
        </div>

        <div className="nlab-screen">
          <div className="nlab-target">
            generated · {pct}% trained {phase === 'done' ? '· sampled (temp 0.5)' : ''}
          </div>
          <div className="nlab-out shakespeare-out">
            {sample || (phase === 'training' ? '' : '…')}
            {phase === 'training' && <span className="nlab-caret" />}
          </div>
          <div className="nlab-tag">
            {phase === 'idle'
              ? 'idle — press train on GPU'
              : phase === 'done'
                ? 'done · trained on WebGPU'
                : 'training a 4-head transformer… (sample on completion)'}
          </div>
        </div>
      </div>
    </section>
  );
}
