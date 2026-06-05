import { useEffect, useRef, useState } from 'react';
import { fetchMnistRaw } from '../data/mnist';
import { CpuMlp, gatherBatchCpu } from '../nn/cpu';
import { crossEntropyCpu, accuracy } from '../nn/reference';
import { mulberry32, gaussian } from '../data/synthetic';

const STEPS = 240;
const HIDDEN = 32;
const BATCH = 32;
const LR = 0.01;

interface Sample {
  pixels: Uint8Array;
  pred: number;
  label: number;
}

export default function CpuTrainer({ forced }: { forced: boolean }) {
  const [phase, setPhase] = useState<'loading' | 'training' | 'done' | 'error'>('loading');
  const [status, setStatus] = useState('starting CPU engine…');
  const [lossHist, setLossHist] = useState<number[]>([]);
  const [metrics, setMetrics] = useState({ step: 0, loss: NaN, trainAcc: 0, testAcc: NaN, stepsPerSec: 0 });
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    let alive = true;
    (async () => {
      let data;
      try {
        data = await fetchMnistRaw((msg) => {
          if (alive) setStatus(msg);
        });
      } catch (e) {
        if (alive) {
          setStatus(e instanceof Error ? e.message : String(e));
          setPhase('error');
        }
        return;
      }
      if (!alive) return;
      setPhase('training');

      const D0 = data.pixels;
      const C = 10;
      const rng = mulberry32(7);
      const he = (fanIn: number, len: number) => {
        const s = Math.sqrt(2 / fanIn);
        const o = new Float32Array(len);
        for (let i = 0; i < len; i++) o[i] = gaussian(rng) * s;
        return o;
      };
      const mlp = new CpuMlp(D0, HIDDEN, C, BATCH, {
        W1: he(D0, D0 * HIDDEN),
        b1: new Float32Array(HIDDEN),
        W2: he(HIDDEN, HIDDEN * C),
        b2: new Float32Array(C),
      });
      const idx = new Uint32Array(BATCH);
      const lab = new Uint32Array(BATCH);
      const hist: number[] = [];
      let step = 0;

      const evalTest = (): number => {
        const batches = Math.min(32, Math.floor(data!.test.count / BATCH));
        let acc = 0;
        for (let tb = 0; tb < batches; tb++) {
          for (let i = 0; i < BATCH; i++) {
            const id = tb * BATCH + i;
            idx[i] = id;
            lab[i] = data!.test.labels[id];
          }
          mlp.forward(gatherBatchCpu(data!.test.images, idx, D0));
          acc += accuracy(mlp.probs, lab, BATCH, C);
        }
        return batches > 0 ? acc / batches : 0;
      };

      const finish = () => {
        const testAcc = evalTest();
        for (let i = 0; i < BATCH; i++) {
          idx[i] = i;
          lab[i] = data!.test.labels[i];
        }
        mlp.forward(gatherBatchCpu(data!.test.images, idx, D0));
        const P = mlp.probs;
        const samp: Sample[] = [];
        for (let i = 0; i < 12; i++) {
          let pred = 0;
          for (let c = 1; c < C; c++) if (P[i * C + c] > P[i * C + pred]) pred = c;
          samp.push({
            pixels: data!.test.images.subarray(i * D0, (i + 1) * D0),
            pred,
            label: data!.test.labels[i],
          });
        }
        if (!alive) return;
        setSamples(samp);
        setMetrics((m) => ({ ...m, testAcc }));
        setPhase('done');
      };

      const chunk = () => {
        if (!alive) return;
        const t0 = performance.now();
        const K = 4;
        let lastLoss = NaN;
        let lastAcc = 0;
        for (let k = 0; k < K && step < STEPS; k++) {
          for (let i = 0; i < BATCH; i++) {
            const r = Math.floor(rng() * data!.train.count);
            idx[i] = r;
            lab[i] = data!.train.labels[r];
          }
          mlp.forward(gatherBatchCpu(data!.train.images, idx, D0));
          const losses = crossEntropyCpu(mlp.probs, lab, BATCH, C);
          let s = 0;
          for (let i = 0; i < BATCH; i++) s += losses[i];
          lastLoss = s / BATCH;
          lastAcc = accuracy(mlp.probs, lab, BATCH, C);
          mlp.backward(lab);
          mlp.step(LR);
          hist.push(lastLoss);
          step += 1;
        }
        const sps = K / ((performance.now() - t0) / 1000);
        setLossHist(hist.slice());
        setMetrics({ step, loss: lastLoss, trainAcc: lastAcc, testAcc: NaN, stepsPerSec: sps });
        if (step < STEPS) setTimeout(chunk, 0);
        else finish();
      };
      chunk();
    })();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="card">
      <h2>
        <span className="ch">CPU</span>CPU fallback · training without WebGPU
      </h2>
      <div className="banner info" style={{ marginBottom: 16 }}>
        {forced ? 'Forced CPU mode (?cpu=1). ' : 'WebGPU unavailable — running on the CPU. '}
        Same gradient-checked math, in plain JS — slower, but it runs everywhere.
      </div>

      {phase === 'loading' && <p className="banner info">{status}</p>}
      {phase === 'error' && (
        <p className="banner fail">
          <code className="inline">{status}</code>
        </p>
      )}

      {(phase === 'training' || phase === 'done') && (
        <>
          <div className="metrics-grid">
            <Stat label="step" value={`${metrics.step} / ${STEPS}`} />
            <Stat label="loss" value={Number.isFinite(metrics.loss) ? metrics.loss.toFixed(3) : '—'} accent="amber" />
            <Stat label="train acc" value={`${(metrics.trainAcc * 100).toFixed(1)}%`} />
            <Stat
              label="test acc"
              value={Number.isFinite(metrics.testAcc) ? `${(metrics.testAcc * 100).toFixed(1)}%` : '…'}
              accent="green"
            />
            <Stat label="steps/s" value={`${Math.round(metrics.stepsPerSec)}`} accent="blue" />
          </div>
          <div style={{ marginTop: 16 }}>
            <Spark data={lossHist} />
            <div className="spark-cap">cross-entropy · {STEPS} steps · 784→{HIDDEN}→10 · CPU (JS)</div>
          </div>
          {samples.length > 0 && (
            <>
              <div className="group-label" style={{ marginTop: 22 }}>
                ▌ sample predictions · CPU inference
              </div>
              <div className="cpu-samples">
                {samples.map((s, i) => (
                  <DigitThumb key={i} pixels={s.pixels} pred={s.pred} label={s.label} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: 'amber' | 'green' | 'blue' }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${accent ? ' ' + accent : ''}`}>{value}</div>
    </div>
  );
}

function Spark({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const w = 600;
  const h = 72;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => `${((i / (data.length - 1)) * w).toFixed(1)},${(h - 4 - ((v - min) / range) * (h - 8)).toFixed(1)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="spark" style={{ width: '100%', height: h }}>
      <polyline points={pts} fill="none" stroke="var(--pink)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

function DigitThumb({ pixels, pred, label }: { pixels: Uint8Array; pred: number; label: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
      const v = pixels[i] ?? 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  const ok = pred === label;
  return (
    <div className={`cpu-sample${ok ? '' : ' wrong'}`}>
      <canvas ref={ref} width={28} height={28} />
      <span>{ok ? pred : `${pred}≠${label}`}</span>
    </div>
  );
}
