// nano-GPT increment 4: the interactive "watch it learn to write" panel.
// Trains the char transformer live on the CPU (the gradient-checked engine from
// nanogpt.ts) in cooperative setTimeout chunks so the UI stays responsive, and
// streams the loss curve + the greedily-generated text as it sharpens from
// noise into the target sentence. No GPU required — runs in every browser.

import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt } from '../nn/nanogpt';
import { mulberry32 } from '../data/synthetic';

const TEXT = 'the quick brown fox jumps over the lazy dog. ';
const PROMPT = 'the q';
const GEN = 44;
const T = 20;
const STEPS = 700;
const CHUNK = 14;
const LR = 0.01;

type Phase = 'idle' | 'training' | 'done';

export function NanoGptLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(NaN);
  const [sample, setSample] = useState('');
  const lossHist = useRef<number[]>([]);
  const [, force] = useState(0);
  const running = useRef(false);

  const stop = useCallback(() => {
    running.current = false;
  }, []);

  const train = useCallback(() => {
    running.current = false; // cancel any in-flight run
    const tok = new CharTokenizer(TEXT);
    const data = tok.encode(TEXT + TEXT);
    const cfg = { vocab: tok.vocab, dEmbed: 32, dFF: 64, blockSize: T };
    const model = new NanoGpt(cfg, 3);
    const ps = model.params();
    const mv = ps.map((p) => new Float32Array(p.w.length));
    const vv = ps.map((p) => new Float32Array(p.w.length));
    const rng = mulberry32(5);

    const corpusLoss = (): number => {
      let sum = 0;
      let n = 0;
      for (let s = 0; s + T + 1 <= data.length; s += 2) {
        sum += NanoGpt.crossEntropy(
          model.forward(data.slice(s, s + T)),
          data.slice(s + 1, s + T + 1),
          T,
          cfg.vocab,
        ).loss;
        n++;
      }
      return sum / n;
    };
    const generate = (): string =>
      tok.decode(model.generate(tok.encode(PROMPT), GEN, 0, mulberry32(0)));

    lossHist.current = [corpusLoss()];
    setPhase('training');
    setStep(0);
    setLoss(lossHist.current[0]);
    setSample(generate());
    running.current = true;
    let s = 0;

    const loop = (): void => {
      if (!running.current) return;
      for (let k = 0; k < CHUNK && s < STEPS; k++) {
        s++;
        const start = Math.floor(rng() * (data.length - T - 1));
        const ids = data.slice(start, start + T);
        const targets = data.slice(start + 1, start + T + 1);
        model.zeroGrad();
        model.backward(model.forwardCache(ids), targets);
        const bc1 = 1 - Math.pow(0.9, s);
        const bc2 = 1 - Math.pow(0.999, s);
        for (let pi = 0; pi < ps.length; pi++) {
          const w = ps[pi].w;
          const g = ps[pi].g;
          const m = mv[pi];
          const v = vv[pi];
          for (let i = 0; i < w.length; i++) {
            m[i] = 0.9 * m[i] + 0.1 * g[i];
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
            w[i] -= (LR * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + 1e-8);
          }
        }
      }
      lossHist.current.push(corpusLoss());
      setStep(s);
      setLoss(lossHist.current[lossHist.current.length - 1]);
      setSample(generate());
      force((x) => x + 1);
      if (s < STEPS) {
        setTimeout(loop, 0);
      } else {
        running.current = false;
        setPhase('done');
      }
    };
    setTimeout(loop, 0);
  }, []);

  const reset = useCallback(() => {
    running.current = false;
    lossHist.current = [];
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setSample('');
  }, []);

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
    .map((l, i) => {
      const x = (i / Math.max(1, hist.length - 1)) * 100;
      const y = 100 - ((l - min) / (max - min || 1)) * 100;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');

  const pct = Math.round((step / STEPS) * 100);

  return (
    <section className="card nlab" id="nanogpt">
      <h2>
        <span className="ch">CH5</span>nano-GPT · watch it learn to write
        <span className="meta">1-layer char transformer · trained live on the CPU</span>
      </h2>

      <div className="nlab-grid">
        <div className="nlab-left">
          <p className="nlab-blurb">
            A from-scratch transformer (causal self-attention + MLP), the same gradient-checked
            engine verified above. Hit train and watch it overfit a sentence — the greedy output
            below morphs from noise into the target as the loss falls.
          </p>
          <div className="nlab-controls">
            {phase === 'training' ? (
              <button className="btn" onClick={stop}>
                stop
              </button>
            ) : (
              <button className="btn primary" onClick={train}>
                {phase === 'done' ? 'train again' : 'train'}
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
              <polyline points={pts} fill="none" stroke="var(--purple)" strokeWidth={2} />
            )}
          </svg>
        </div>

        <div className="nlab-screen">
          <div className="nlab-target">
            target&nbsp;·&nbsp;<span>{TEXT.trim()}</span>
          </div>
          <div className="nlab-out">
            <span className="nlab-prompt">{PROMPT}</span>
            {sample.slice(PROMPT.length)}
            {phase === 'training' && <span className="nlab-caret" />}
          </div>
          <div className="nlab-tag">
            {phase === 'idle'
              ? 'idle — press train'
              : phase === 'done'
                ? 'converged · greedy decode'
                : 'learning…'}
          </div>
        </div>
      </div>
    </section>
  );
}
