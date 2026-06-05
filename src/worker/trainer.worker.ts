/// <reference lib="webworker" />
import { getGpuContext } from '../gpu/device';
import { GpuTensor } from '../gpu/tensor';
import { gather, createU32Buffer } from '../gpu/ops';
import { Mlp } from '../nn/mlp';
import { Adam } from '../nn/optimizer';
import { accuracy } from '../nn/reference';
import { mulberry32, gaussian } from '../data/synthetic';
import { loadMnist, type MnistData } from '../data/mnist';
import { pca2d } from '../nn/pca';
import type { InMsg, OutMsg } from './protocol';

const sw = self as unknown as DedicatedWorkerGlobalScope;
function post(msg: OutMsg, transfer: Transferable[] = []): void {
  sw.postMessage(msg, transfer);
}

const B = 64;
const C = 10;
const CHUNK = 8;

let device: GPUDevice;
let data: MnistData | null = null;
let mlp: Mlp | null = null;
let opt: Adam | null = null;
let running = false;
let infering = false;
let step = 0;
let hiddenDim = 64;
let lr = 0.005;
let chunkCounter = 0;

let batchImages: GpuTensor;
let idxBuf: GPUBuffer;
let labBuf: GPUBuffer;
const idxArr = new Uint32Array(B);
const labArr = new Uint32Array(B);
let rng = mulberry32(1);

let probeNorm = new Float32Array(0);
let probeLabel = 0;

function buildModel(): void {
  mlp?.destroy();
  opt?.destroy();
  rng = mulberry32(1);
  const P = data!.pixels;
  const heInit = (fanIn: number, len: number): Float32Array => {
    const std = Math.sqrt(2 / fanIn);
    const o = new Float32Array(len);
    for (let i = 0; i < len; i++) o[i] = gaussian(rng) * std;
    return o;
  };
  mlp = new Mlp(
    device,
    { inputDim: P, hiddenDim, outputDim: C, batch: B },
    {
      W1: heInit(P, P * hiddenDim),
      b1: new Float32Array(hiddenDim),
      W2: heInit(hiddenDim, hiddenDim * C),
      b2: new Float32Array(C),
    },
  );
  opt = new Adam(device, mlp.params(), { lr });
  step = 0;
}

async function init(): Promise<void> {
  const ctx = await getGpuContext();
  device = ctx.device;
  post({ type: 'status', message: 'fetching MNIST…' });
  data = await loadMnist(device, (m) => post({ type: 'status', message: m }));

  batchImages = GpuTensor.zeros(device, [B, data.pixels], { label: 'batch' });
  idxBuf = device.createBuffer({
    label: 'idx',
    size: B * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  labBuf = createU32Buffer(device, labArr, 'lab');

  const px = data.pixels;
  probeNorm = new Float32Array(px);
  for (let i = 0; i < px; i++) probeNorm[i] = data.test.images[i] / 255;
  probeLabel = data.test.labels[0];

  buildModel();
  post({
    type: 'ready',
    adapter: ctx.label,
    train: data.train.count,
    test: data.test.count,
    pixels: data.pixels,
    hidden: hiddenDim,
    lr,
  });
  await postActivations();
  await postEval();
}

function sampleTrainBatch(): void {
  for (let i = 0; i < B; i++) {
    const r = Math.floor(rng() * data!.train.count);
    idxArr[i] = r;
    labArr[i] = data!.train.labels[r];
  }
  device.queue.writeBuffer(idxBuf, 0, idxArr);
  device.queue.writeBuffer(labBuf, 0, labArr);
  gather(device, data!.train.packed, idxBuf, B, data!.pixels, batchImages);
}

async function trainChunk(): Promise<void> {
  if (!running || !mlp || !opt || !data) return;
  if (infering) {
    setTimeout(trainChunk, 0);
    return;
  }
  const t0 = performance.now();
  let loss = 0;
  for (let k = 0; k < CHUNK; k++) {
    sampleTrainBatch();
    loss = await mlp.forwardLoss(batchImages, labBuf);
    mlp.backward(labBuf);
    opt.step(mlp.grads());
    step++;
  }
  const dt = (performance.now() - t0) / 1000;
  const trainAcc = accuracy(await mlp.P.toArray(), labArr, B, C);
  post({ type: 'metrics', step, loss, trainAcc, stepsPerSec: CHUNK / dt });

  chunkCounter++;
  if (chunkCounter % 2 === 0) await postActivations();
  if (chunkCounter % 25 === 0) await postEval();

  if (running) setTimeout(trainChunk, 0);
}

async function postActivations(): Promise<void> {
  if (!mlp || !data) return;
  device.queue.writeBuffer(batchImages.buffer, 0, probeNorm as Float32Array<ArrayBuffer>);
  mlp.forward(batchImages);
  const [a1, p] = await Promise.all([mlp.A1.toArray(), mlp.P.toArray()]);
  const hidden = a1.slice(0, hiddenDim);
  const probs = p.slice(0, C);
  let pred = 0;
  for (let j = 1; j < C; j++) if (probs[j] > probs[pred]) pred = j;
  const input = probeNorm.slice();
  post({ type: 'activations', input, hidden, probs, label: probeLabel, pred }, [
    input.buffer,
    hidden.buffer,
    probs.buffer,
  ]);
}

async function postEval(): Promise<void> {
  if (!mlp || !data) return;
  const H = hiddenDim;
  const batches = Math.min(8, Math.floor(data.test.count / B));
  const N = batches * B;
  const acts = new Float32Array(N * H);
  const labels = new Uint8Array(N);
  let accSum = 0;

  for (let tb = 0; tb < batches; tb++) {
    for (let i = 0; i < B; i++) {
      const idx = tb * B + i;
      idxArr[i] = idx;
      labArr[i] = data.test.labels[idx];
    }
    device.queue.writeBuffer(idxBuf, 0, idxArr);
    gather(device, data.test.packed, idxBuf, B, data.pixels, batchImages);
    mlp.forward(batchImages);
    const [p, a1] = await Promise.all([mlp.P.toArray(), mlp.A1.toArray()]);
    accSum += accuracy(p, labArr, B, C);
    for (let i = 0; i < B; i++) {
      const row = tb * B + i;
      labels[row] = labArr[i];
      const dst = row * H;
      const src = i * H;
      for (let h = 0; h < H; h++) acts[dst + h] = a1[src + h];
    }
  }

  post({ type: 'testacc', step, testAcc: accSum / batches });
  const coords = pca2d(acts, N, H);
  post({ type: 'embedding', coords, labels, step }, [coords.buffer, labels.buffer]);
}

async function handleInfer(pixels: Float32Array): Promise<void> {
  if (!mlp || !data || pixels.length !== data.pixels) return;
  infering = true;
  try {
    device.queue.writeBuffer(batchImages.buffer, 0, pixels as Float32Array<ArrayBuffer>);
    mlp.forward(batchImages);
    const p = await mlp.P.toArray();
    const probs = p.slice(0, C);
    post({ type: 'probs', probs }, [probs.buffer]);
  } finally {
    infering = false;
  }
}

async function computeLandscape(): Promise<void> {
  if (!mlp || !data) return;
  infering = true;
  try {
    const G = 21;
    const range = 1.0;
    const params = mlp.params();
    const W0 = await Promise.all(params.map((p) => p.toArray()));
    const rng2 = mulberry32(step + 17);
    const makeDir = () =>
      W0.map((w) => {
        const d = new Float32Array(w.length);
        let wn = 0;
        let dn = 0;
        for (let i = 0; i < w.length; i++) {
          const g = gaussian(rng2);
          d[i] = g;
          dn += g * g;
          wn += w[i] * w[i];
        }
        const scale = Math.sqrt(wn) / (Math.sqrt(dn) || 1);
        for (let i = 0; i < w.length; i++) d[i] *= scale;
        return d;
      });
    const d1 = makeDir();
    const d2 = makeDir();

    for (let i = 0; i < B; i++) {
      idxArr[i] = i;
      labArr[i] = data.train.labels[i];
    }
    device.queue.writeBuffer(idxBuf, 0, idxArr);
    device.queue.writeBuffer(labBuf, 0, labArr);
    gather(device, data.train.packed, idxBuf, B, data.pixels, batchImages);

    const grid = new Float32Array(G * G);
    const tmp = params.map((_, k) => new Float32Array(W0[k].length));
    for (let gi = 0; gi < G; gi++) {
      const a = (gi / (G - 1)) * 2 * range - range;
      for (let gj = 0; gj < G; gj++) {
        const b = (gj / (G - 1)) * 2 * range - range;
        for (let k = 0; k < params.length; k++) {
          const w0 = W0[k];
          const e1 = d1[k];
          const e2 = d2[k];
          const t = tmp[k];
          for (let i = 0; i < t.length; i++) t[i] = w0[i] + a * e1[i] + b * e2[i];
          device.queue.writeBuffer(params[k].buffer, 0, t);
        }
        grid[gi * G + gj] = await mlp.forwardLoss(batchImages, labBuf);
      }
    }
    for (let k = 0; k < params.length; k++)
      device.queue.writeBuffer(params[k].buffer, 0, W0[k] as Float32Array<ArrayBuffer>);

    post({ type: 'landscape', grid, size: G, step }, [grid.buffer]);
  } finally {
    infering = false;
  }
}

sw.onmessage = (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      init().catch((err) => post({ type: 'error', message: String(err?.message ?? err) }));
      break;
    case 'start':
      if (!running && mlp) {
        running = true;
        void trainChunk();
      }
      break;
    case 'pause':
      running = false;
      break;
    case 'reset':
      hiddenDim = msg.hidden;
      lr = msg.lr;
      buildModel();
      post({ type: 'metrics', step: 0, loss: NaN, trainAcc: 0, stepsPerSec: 0 });
      void postActivations();
      break;
    case 'infer':
      void handleInfer(msg.pixels);
      break;
    case 'landscape':
      void computeLandscape();
      break;
  }
};
