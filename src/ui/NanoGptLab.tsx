import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt } from '../nn/nanogpt';
import { mulberry32 } from '../data/synthetic';

const DEFAULT_TEXT = 'the quick brown fox jumps over the lazy dog.';
const MAX_LEN = 56;
const T = 20;
const STEPS = 700;
const CHUNK = 14;
const LR = 0.01;
const SEED_LEN = 5;

type Phase = 'idle' | 'training' | 'done';

export function NanoGptLab() {
  const [text, setText] = useState(DEFAULT_TEXT);
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

  const reset = useCallback(() => {
    running.current = false;
    lossHist.current = [];
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setSample('');
  }, []);

  const onText = useCallback(
    (v: string) => {
      setText(v);
      reset();
    },
    [reset],
  );

  const train = useCallback(() => {
    running.current = false;
    const sentence = text.trim() || DEFAULT_TEXT;
    const base = sentence + ' ';
    let corpusText = base;
    while (corpusText.length < 3 * T + 1) corpusText += base;

    const tok = new CharTokenizer(corpusText);
    const data = tok.encode(corpusText);
    const cfg = { vocab: tok.vocab, dEmbed: 32, dFF: 64, blockSize: T };
    const model = new NanoGpt(cfg, 3);
    const ps = model.params();
    const mv = ps.map((p) => new Float32Array(p.w.length));
    const vv = ps.map((p) => new Float32Array(p.w.length));
    const rng = mulberry32(5);

    const seed = sentence.slice(0, Math.min(SEED_LEN, sentence.length));
    const genLen = Math.min(80, sentence.length + 10);

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
      return n ? sum / n : NaN;
    };
    const generate = (): string =>
      tok.decode(model.generate(tok.encode(seed), genLen, 0, mulberry32(0)));

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
  }, [text]);

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
  const target = text.trim() || DEFAULT_TEXT;
  const prompt = target.slice(0, Math.min(SEED_LEN, target.length));

  return (
    <section className="card nlab" id="nanogpt">
      <h2>
        <span className="ch">CH5</span>nano-GPT · watch it learn to write
        <span className="meta">1-layer char transformer · trained live on the CPU</span>
      </h2>

      <div className="nlab-grid">
        <div className="nlab-left">
          <label className="nlab-field">
            <span className="nlab-label">target sentence</span>
            <input
              className="nlab-input"
              type="text"
              value={text}
              maxLength={MAX_LEN}
              spellCheck={false}
              placeholder={DEFAULT_TEXT}
              onChange={(e) => onText(e.target.value)}
            />
          </label>
          <p className="nlab-reqs">
            Type a sentence — the net memorizes it and writes it back. Lowercase and ≤ {MAX_LEN}{' '}
            characters work best; it's a tiny 1-layer transformer trained for {STEPS} steps, then
            seeded greedily with the first {SEED_LEN} characters.
          </p>
          <div className="nlab-controls">
            {phase === 'training' ? (
              <button className="btn" onClick={stop}>
                stop
              </button>
            ) : (
              <button className="btn primary" onClick={train} disabled={!text.trim()}>
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
            target&nbsp;·&nbsp;<span>{target}</span>
          </div>
          <div className="nlab-out">
            <span className="nlab-prompt">{prompt}</span>
            {sample.slice(prompt.length)}
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
