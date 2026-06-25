import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt } from '../nn/nanogpt';
import { GpuNanoGpt } from '../nn/nanogpt-gpu';
import { getGpuContext } from '../gpu/device';
import { mulberry32 } from '../data/synthetic';
import { ember } from './ember';

const SHAKESPEARE = `to be, or not to be, that is the question:
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles. `;

const T = 32;
const STEPS = 800;
const CHUNK = 5;
const LR = 0.003;
const GEN = 180;
const MIN_CHARS = 24;
const MAX_CHARS = 4000;

type Phase = 'idle' | 'training' | 'done';

function drawAttn(canvas: HTMLCanvasElement | null, data: Float32Array | null, n: number): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const px = Math.max(1, Math.round(canvas.clientWidth * dpr));
  if (canvas.width !== px) canvas.width = px;
  if (canvas.height !== px) canvas.height = px;
  const W = canvas.width;
  ctx.clearRect(0, 0, W, W);
  ctx.fillStyle = '#090d18';
  ctx.fillRect(0, 0, W, W);
  if (!data) return;
  let max = 1e-6;
  for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
  const matW = W * 0.84;
  const gutterX = W * 0.87;
  const gutterW = W - gutterX;
  const cellW = matW / n;
  const cellH = W / n;
  for (let i = 0; i < n; i++) {
    let rowMax = 0;
    for (let j = 0; j <= i; j++) {
      const v = data[i * n + j] / max;
      if (v > rowMax) rowMax = v;
      const [r, g, b] = ember(v);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
    }
    const [gr, gg, gb] = ember(rowMax);
    ctx.fillStyle = `rgb(${gr},${gg},${gb})`;
    ctx.fillRect(gutterX, i * cellH, rowMax * gutterW, cellH + 0.5);
  }
}

export function ShakespeareLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(NaN);
  const [sample, setSample] = useState('');
  const [corpus, setCorpus] = useState(SHAKESPEARE);
  const lossHist = useRef<number[]>([]);
  const attnRef = useRef<Float32Array | null>(null);
  const attnCanvas = useRef<HTMLCanvasElement>(null);
  const running = useRef(false);
  const gpuRef = useRef<GpuNanoGpt | null>(null);
  const [, force] = useState(0);

  const tooShort = corpus.trim().length < MIN_CHARS;

  const stop = useCallback(() => {
    running.current = false;
    setPhase('done');
  }, []);

  const reset = useCallback(() => {
    running.current = false;
    lossHist.current = [];
    attnRef.current = null;
    drawAttn(attnCanvas.current, null, T);
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setSample('');
  }, []);

  const train = useCallback(() => {
    const text = corpus.trim().slice(0, MAX_CHARS);
    if (text.length < MIN_CHARS) return;
    running.current = false;
    lossHist.current = [];
    attnRef.current = null;
    setPhase('training');
    setStep(0);
    setLoss(NaN);
    setSample('');
    void (async () => {
      const { device } = await getGpuContext();
      const tok = new CharTokenizer(text);
      const reps = Math.max(2, Math.ceil((T * 4) / text.length));
      const data = tok.encode(text.repeat(reps));
      const cfg = { vocab: tok.vocab, dEmbed: 64, dFF: 128, blockSize: T };
      const cpu = new NanoGpt(cfg, 7);
      const gpu = new GpuNanoGpt(device, cpu);
      gpuRef.current = gpu;
      const rng = mulberry32(11);
      const probe = data.slice(0, T);
      const seed = text.slice(0, 10);
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
        lossHist.current.push(l);
        await gpu.syncTo(cpu);
        attnRef.current = await gpu.attention(probe);
        drawAttn(attnCanvas.current, attnRef.current, T);
        setStep(s);
        setLoss(l);
        setSample(tok.decode(cpu.generate(tok.encode(seed), GEN, 0.5, mulberry32(0))));
        force((x) => x + 1);
        if (s < STEPS && running.current) {
          setTimeout(loop, 0);
        } else {
          gpu.destroy();
          gpuRef.current = null;
          if (s >= STEPS) setPhase('done');
        }
      };
      setTimeout(loop, 0);
    })();
  }, [corpus]);

  useEffect(
    () => () => {
      running.current = false;
    },
    [],
  );

  const hist = lossHist.current;
  const max = hist.length ? Math.max(...hist) : 1;
  const min = hist.length ? Math.min(...hist) : 0;
  const pts = hist
    .map((v, i) => {
      const x = (i / Math.max(1, hist.length - 1)) * 100;
      const y = 100 - ((v - min) / (max - min || 1)) * 100;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  const pct = Math.round((step / STEPS) * 100);

  return (
    <section className="card nlab" id="shakespeare">
      <h2>
        <span className="ch">CH6</span>nano-GPT · learns to write, live on the GPU
        <span className="meta">trained on WebGPU · forward + backward + Adam · on your text</span>
      </h2>

      <div className="nlab-grid">
        <div className="nlab-left">
          <p className="nlab-reqs">
            A from-scratch transformer trained <b>entirely on your GPU</b> — forward, backprop and
            Adam all in WGSL. It ships learning a passage of Hamlet, but the corpus is yours: paste
            lyrics, code, your bio — anything. Watch the output go from noise to your words, and the
            attention matrix sharpen, as the loss falls.
          </p>
          <div className="corpus-field">
            <div className="corpus-label">
              <span>corpus · edit to train on your own text</span>
              <span className={tooShort ? 'corpus-count low' : 'corpus-count'}>
                {corpus.trim().length} chars
              </span>
            </div>
            <textarea
              className="corpus-area"
              value={corpus}
              spellCheck={false}
              disabled={phase === 'training'}
              onChange={(e) => setCorpus(e.target.value)}
            />
            {tooShort && (
              <span className="corpus-hint">give it at least {MIN_CHARS} characters to chew on</span>
            )}
          </div>
          <div className="nlab-controls">
            {phase === 'training' ? (
              <button className="btn" onClick={stop}>
                stop
              </button>
            ) : (
              <button className="btn primary" onClick={train} disabled={tooShort}>
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
              <b>{pct}%</b>
              <span>progress</span>
            </div>
          </div>
          <svg className="nlab-spark" viewBox="0 0 100 100" preserveAspectRatio="none">
            {hist.length > 1 && (
              <>
                <polygon points={`0,100 ${pts} 100,100`} fill="rgba(255,176,64,0.16)" />
                <polyline points={pts} fill="none" stroke="var(--purple)" strokeWidth={2} />
              </>
            )}
          </svg>
          <div className="attn-wrap">
            <span className="attn-label">
              attention · {T}×{T} · token i attends to j ≤ i
              <span className="ember-key" />
            </span>
            <canvas ref={attnCanvas} className="attn-canvas" />
          </div>
        </div>

        <div className="nlab-screen">
          <div className="nlab-target">generated · sampled (temp 0.5)</div>
          <div className="nlab-out shakespeare-out">
            {sample || '…'}
            {phase === 'training' && <span className="nlab-caret" />}
          </div>
          <div className="nlab-tag">
            {phase === 'idle'
              ? 'idle — press train on GPU'
              : phase === 'done'
                ? 'done · trained on WebGPU'
                : 'learning your text…'}
          </div>
        </div>
      </div>
    </section>
  );
}
