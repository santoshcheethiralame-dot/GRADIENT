import { useCallback, useEffect, useRef, useState } from 'react';
import { NanoGpt } from '../nn/nanogpt';
import { BpeTokenizer } from '../nn/bpe';
import { nanoGptGpuBackward } from '../nn/nanogpt-gpu';
import { getGpuContext } from '../gpu/device';
import { mulberry32 } from '../data/synthetic';
import { ember } from './ember';

const CORPUS = `to be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles
and by opposing end them. to die, to sleep,
no more; and by a sleep to say we end
the heart-ache and the thousand natural shocks
that flesh is heir to. to sleep, perchance to dream:
ay, there's the rub, for in that sleep of death
what dreams may come, when we have shuffled off
this mortal coil, must give us pause.

shall i compare thee to a summer's day?
thou art more lovely and more temperate.
rough winds do shake the darling buds of may,
and summer's lease hath all too short a date.
sometime too hot the eye of heaven shines,
and often is his gold complexion dimm'd;
and every fair from fair sometime declines,
by chance, or nature's changing course, untrimm'd. `;

const T = 48;
const STEPS = 400;
const CHUNK = 2;
const LR = 0.0025;
const GEN = 80;
const VOCAB_TARGET = 96;
const D_EMBED = 96;
const D_FF = 192;
const N_HEADS = 3;
const N_BLOCKS = 2;

const B1 = 0.9;
const B2 = 0.999;
const EPS = 1e-8;

type Phase = 'idle' | 'training' | 'done';

function drawStrip(canvas: HTMLCanvasElement | null, hist: number[]): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const H = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#070a12';
  ctx.fillRect(0, 0, W, H);
  if (hist.length < 1) return;
  let max = 1e-6;
  for (const v of hist) if (v > max) max = v;
  const cw = W / hist.length;
  for (let i = 0; i < hist.length; i++) {
    const [r, g, b] = ember(hist[i] / max);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i * cw, 0, cw + 0.6, H);
  }
}

export function DeepLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(NaN);
  const [toks, setToks] = useState(0);
  const [sample, setSample] = useState('');
  const [vocabInfo, setVocabInfo] = useState('');
  const running = useRef(false);
  const lossHist = useRef<number[]>([]);
  const stripRef = useRef<HTMLCanvasElement>(null);

  const stop = useCallback(() => {
    running.current = false;
    setPhase('done');
  }, []);

  const reset = useCallback(() => {
    running.current = false;
    lossHist.current = [];
    drawStrip(stripRef.current, []);
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setToks(0);
    setSample('');
  }, []);

  const train = useCallback(() => {
    running.current = false;
    lossHist.current = [];
    setPhase('training');
    setStep(0);
    setLoss(NaN);
    setToks(0);
    setSample('');
    void (async () => {
      const { device } = await getGpuContext();
      const tok = new BpeTokenizer(CORPUS, VOCAB_TARGET);
      const data = tok.encode(CORPUS);
      setVocabInfo(`${tok.vocab} BPE tokens · ${tok.merges} merges`);
      const cfg = {
        vocab: tok.vocab,
        dEmbed: D_EMBED,
        dFF: D_FF,
        blockSize: T,
        nHeads: N_HEADS,
        nBlocks: N_BLOCKS,
      };
      const model = new NanoGpt(cfg, 7);
      const mom = new Map(
        model.params().map((p) => [
          p.name,
          { m: new Float32Array(p.w.length), v: new Float32Array(p.w.length) },
        ]),
      );
      const rng = mulberry32(11);
      const seed = data.slice(0, 8);
      const t0 = performance.now();
      running.current = true;
      let s = 0;
      let adamT = 0;
      const loop = async (): Promise<void> => {
        if (!running.current) return;
        let l = NaN;
        for (let c = 0; c < CHUNK && s < STEPS; c++) {
          s++;
          const start = Math.floor(rng() * (data.length - T - 1));
          const ids = data.slice(start, start + T);
          const targets = data.slice(start + 1, start + T + 1);
          const grads = await nanoGptGpuBackward(device, model, ids, targets);
          adamT++;
          const bc1 = 1 - Math.pow(B1, adamT);
          const bc2 = 1 - Math.pow(B2, adamT);
          for (const p of model.params()) {
            const g = grads[p.name];
            if (!g) continue;
            const mo = mom.get(p.name);
            if (!mo) continue;
            for (let i = 0; i < p.w.length; i++) {
              mo.m[i] = B1 * mo.m[i] + (1 - B1) * g[i];
              mo.v[i] = B2 * mo.v[i] + (1 - B2) * g[i] * g[i];
              p.w[i] -= (LR * (mo.m[i] / bc1)) / (Math.sqrt(mo.v[i] / bc2) + EPS);
            }
          }
          const la = grads['__loss'];
          if (la) {
            let sum = 0;
            for (let i = 0; i < la.length; i++) sum += la[i];
            l = sum / la.length;
          }
        }
        if (!Number.isNaN(l)) lossHist.current.push(l);
        drawStrip(stripRef.current, lossHist.current);
        setStep(s);
        setLoss(l);
        setToks(Math.round((s * T) / ((performance.now() - t0) / 1000)));
        if (s < STEPS && running.current) {
          setTimeout(loop, 0);
        } else {
          setSample(tok.decode(model.generate(seed.slice(), GEN, 0.5, mulberry32(0))));
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
    <section className="card nlab" id="deep">
      <h2>
        <span className="ch">CH10</span>deep transformer — stacked blocks + BPE
        <span className="meta">
          {N_BLOCKS} blocks · {N_HEADS} heads · byte-pair tokens
        </span>
      </h2>

      <div className="race-body">
        <p className="nlab-reqs">
          The full stack, scaled in depth and tokens: a <b>{N_BLOCKS}-block</b>, {N_HEADS}-head
          transformer over a real <b>byte-pair-encoding</b> vocabulary, trained on the GPU. Forward
          and backprop run in WGSL (the gradients are the same ones the self-test diffs against the
          CPU and PyTorch); Adam runs on the read-back grads. {vocabInfo && <b>{vocabInfo}.</b>}
        </p>

        <div className="nlab-stats">
          <div>
            <b>{step}</b>
            <span>/ {STEPS} steps</span>
          </div>
          <div>
            <b>{Number.isNaN(loss) ? '—' : loss.toFixed(3)}</b>
            <span>loss (nats/token)</span>
          </div>
          <div>
            <b>{toks ? toks.toLocaleString() : '—'}</b>
            <span>tokens / sec</span>
          </div>
          <div>
            <b>{pct}%</b>
            <span>progress</span>
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
                : 'training a 2-block BPE transformer… (sample on completion)'}
          </div>
        </div>

        <div className="conf-strip-wrap">
          <span className="conf-label">
            loss per step · warm = high, cools as it learns
            <span className="ember-key" />
          </span>
          <canvas ref={stripRef} className="conf-strip" />
        </div>

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
      </div>
    </section>
  );
}
