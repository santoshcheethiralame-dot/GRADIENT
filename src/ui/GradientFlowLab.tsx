import { useCallback, useEffect, useRef, useState } from 'react';
import { CharTokenizer, NanoGpt, NANO_CORPUS } from '../nn/nanogpt';
import { GpuNanoGpt, nanoGptGpuBackward } from '../nn/nanogpt-gpu';
import { getGpuContext } from '../gpu/device';
import { mulberry32 } from '../data/synthetic';

const T = 32;
const STEPS = 600;
const CHUNK = 4;
const LR = 0.003;

const LAYERS: { label: string; keys: string[] }[] = [
  { label: 'LM head', keys: ['head'] },
  { label: 'final norm', keys: ['lnf.g', 'lnf.b'] },
  { label: 'MLP · down', keys: ['Wff2', 'bff2'] },
  { label: 'MLP · up', keys: ['Wff1', 'bff1'] },
  { label: 'norm 2', keys: ['ln2.g', 'ln2.b'] },
  { label: 'attn · out', keys: ['Wo'] },
  { label: 'attn · Q K V', keys: ['Wq', 'Wk', 'Wv'] },
  { label: 'norm 1', keys: ['ln1.g', 'ln1.b'] },
  { label: 'pos embed', keys: ['posEmb'] },
  { label: 'tok embed', keys: ['tokEmb'] },
];

type Phase = 'idle' | 'running' | 'done';
type Bar = { label: string; rms: number; rel: number };

const EMPTY: Bar[] = LAYERS.map((l) => ({ label: l.label, rms: 0, rel: 0 }));

function groupRms(grads: Record<string, Float32Array>, keys: string[]): number {
  let sum = 0;
  let n = 0;
  for (const k of keys) {
    const a = grads[k];
    if (!a) continue;
    for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
    n += a.length;
  }
  return n ? Math.sqrt(sum / n) : 0;
}

export function GradientFlowLab() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(NaN);
  const [bars, setBars] = useState<Bar[]>(EMPTY);
  const running = useRef(false);
  const gpuRef = useRef<GpuNanoGpt | null>(null);

  const stop = useCallback(() => {
    running.current = false;
    setPhase('done');
  }, []);

  const reset = useCallback(() => {
    running.current = false;
    setPhase('idle');
    setStep(0);
    setLoss(NaN);
    setBars(EMPTY);
  }, []);

  const run = useCallback(() => {
    running.current = false;
    setPhase('running');
    setStep(0);
    setLoss(NaN);
    setBars(EMPTY);
    void (async () => {
      const { device } = await getGpuContext();
      const tok = new CharTokenizer(NANO_CORPUS);
      const data = tok.encode(NANO_CORPUS);
      const cfg = { vocab: tok.vocab, dEmbed: 64, dFF: 128, blockSize: T };
      const cpu = new NanoGpt(cfg, 7);
      const gpu = new GpuNanoGpt(device, cpu);
      gpuRef.current = gpu;
      const rng = mulberry32(11);
      const probeIds = data.slice(0, T);
      const probeTargets = data.slice(1, T + 1);
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
        await gpu.syncTo(cpu);
        const grads = await nanoGptGpuBackward(device, cpu, probeIds, probeTargets);
        const rms = LAYERS.map((layer) => groupRms(grads, layer.keys));
        let maxR = 1e-12;
        for (const r of rms) if (r > maxR) maxR = r;
        setBars(LAYERS.map((layer, i) => ({ label: layer.label, rms: rms[i], rel: rms[i] / maxR })));
        setStep(s);
        setLoss(l);
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
  }, []);

  useEffect(
    () => () => {
      running.current = false;
    },
    [],
  );

  return (
    <section className="card nlab" id="gradflow">
      <h2>
        <span className="ch">CH8</span>gradient flow — watch backprop, live
        <span className="meta">|∇| per layer · output → input</span>
      </h2>

      <div className="gflow-body">
        <p className="nlab-reqs">
          Everyone visualizes attention (the forward pass). This shows the <b>backward</b> one: every
          layer's gradient magnitude (RMS of ∂loss/∂w), computed by the same GPU backward that trains
          the net. Bars are ordered the way gradients propagate — from the loss at the <b>LM head</b>{' '}
          (top) back to the <b>embeddings</b> (bottom) — each scaled to that step's largest gradient,
          so you can see <b>where the learning signal concentrates</b> as it flows and pulses each step.
        </p>

        <div className="nlab-stats">
          <div>
            <b>{step}</b>
            <span>/ {STEPS} steps</span>
          </div>
          <div>
            <b>{Number.isNaN(loss) ? '—' : loss.toFixed(3)}</b>
            <span>loss (nats/char)</span>
          </div>
        </div>

        <div className="gflow-screen">
          <div className="gflow-dir">∇ backward — loss enters here ↓</div>
          {bars.map((b) => (
            <div className="gflow-row" key={b.label}>
              <span className="gflow-label">{b.label}</span>
              <div className="gflow-track">
                <div
                  className="gflow-fill"
                  style={{ width: `${(b.rel * 100).toFixed(1)}%`, opacity: 0.32 + 0.68 * b.rel }}
                />
              </div>
              <span className="gflow-val">{b.rms ? b.rms.toExponential(1) : '—'}</span>
            </div>
          ))}
          <div className="gflow-dir bottom">↓ ∇ reaches the inputs</div>
        </div>

        <div className="nlab-controls">
          {phase === 'running' ? (
            <button className="btn" onClick={stop}>
              stop
            </button>
          ) : (
            <button className="btn primary" onClick={run}>
              {phase === 'done' ? 'run again' : 'run backprop'}
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
