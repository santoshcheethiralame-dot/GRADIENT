/// <reference lib="webworker" />
import { NanoGpt } from '../nn/nanogpt';
import { mulberry32 } from '../data/synthetic';
import type { RaceIn, RaceOut, RaceStart } from './race-protocol';

const sw = self as unknown as DedicatedWorkerGlobalScope;
function post(msg: RaceOut): void {
  sw.postMessage(msg);
}

const B1 = 0.9;
const B2 = 0.999;
const EPS = 1e-8;
const SLICE_MS = 60;

let running = false;

function runRace(msg: RaceStart): void {
  running = true;
  const { cfg, seed, data, T, lr, durationMs, reportMs } = msg;
  const model = new NanoGpt(cfg, seed);
  const params = model.params();
  const moments = params.map((p) => ({
    m: new Float32Array(p.w.length),
    v: new Float32Array(p.w.length),
  }));
  const rng = mulberry32(99);
  const t0 = performance.now();
  let step = 0;
  let adamT = 0;
  let lastReport = 0;

  const loop = (): void => {
    if (!running) return;
    let loss = NaN;
    const sliceStart = performance.now();
    do {
      const start = Math.floor(rng() * (data.length - T - 1));
      const ids = data.slice(start, start + T);
      const targets = data.slice(start + 1, start + T + 1);
      model.zeroGrad();
      const cache = model.forwardCache(ids);
      model.backward(cache, targets);
      loss = NanoGpt.crossEntropy(cache.logits, targets, T, cfg.vocab).loss;
      adamT++;
      const bc1 = 1 - Math.pow(B1, adamT);
      const bc2 = 1 - Math.pow(B2, adamT);
      for (let pi = 0; pi < params.length; pi++) {
        const w = params[pi].w;
        const g = params[pi].g;
        const mm = moments[pi].m;
        const vv = moments[pi].v;
        for (let i = 0; i < w.length; i++) {
          mm[i] = B1 * mm[i] + (1 - B1) * g[i];
          vv[i] = B2 * vv[i] + (1 - B2) * g[i] * g[i];
          w[i] -= (lr * (mm[i] / bc1)) / (Math.sqrt(vv[i] / bc2) + EPS);
        }
      }
      step++;
    } while (performance.now() - sliceStart < SLICE_MS && performance.now() - t0 < durationMs);

    const elapsed = performance.now() - t0;
    if (elapsed - lastReport >= reportMs || elapsed >= durationMs) {
      post({ type: 'progress', step, loss, elapsedMs: elapsed });
      lastReport = elapsed;
    }
    if (elapsed >= durationMs) {
      running = false;
      post({ type: 'done', step, loss, elapsedMs: elapsed });
      return;
    }
    setTimeout(loop, 0);
  };
  loop();
}

sw.onmessage = (e: MessageEvent<RaceIn>): void => {
  const msg = e.data;
  if (msg.type === 'stop') {
    running = false;
  } else if (msg.type === 'start') {
    runRace(msg);
  }
};
