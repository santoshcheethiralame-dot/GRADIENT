/// <reference lib="webworker" />
// The training worker. Owns its own GPUDevice (one per context), loads MNIST,
// and runs an interruptible continuous training loop. Training proceeds in small
// chunks scheduled via setTimeout(0) so incoming control messages (pause/reset/
// infer) are processed between chunks. Metrics and a probe sample's activations
// are posted back for the live UI; the UI throttles rendering with rAF.

import { getGpuContext } from '../gpu/device';
import { GpuTensor } from '../gpu/tensor';
import { gather, createU32Buffer } from '../gpu/ops';
import { Mlp } from '../nn/mlp';
import { Adam } from '../nn/optimizer';
import { accuracy } from '../nn/reference';
import { mulberry32, gaussian } from '../data/synthetic';
import { loadMnist, type MnistData } from '../data/mnist';
import type { InMsg, OutMsg } from './protocol';

const sw = self as unknown as DedicatedWorkerGlobalScope;
function post(msg: OutMsg, transfer: Transferable[] = []): void {
  sw.postMessage(msg, transfer);
}

const B = 64;
const C = 10;
const CHUNK = 8; // training steps between yields to the message queue

let device: GPUDevice;
let data: MnistData | null = null;
let mlp: Mlp | null = null;
let opt: Adam | null = null;
let running = false;
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

let probeNorm = new Float32Array(0); // fixed probe image (test #0) for the heatmaps
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
  if (chunkCounter % 25 === 0) await postTestAcc();

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

async function postTestAcc(): Promise<void> {
  if (!mlp || !data) return;
  const batches = Math.min(8, Math.floor(data.test.count / B));
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
    accSum += accuracy(await mlp.P.toArray(), labArr, B, C);
  }
  post({ type: 'testacc', step, testAcc: accSum / batches });
}

async function handleInfer(pixels: Float32Array): Promise<void> {
  if (!mlp || !data || pixels.length !== data.pixels) return;
  device.queue.writeBuffer(batchImages.buffer, 0, pixels as Float32Array<ArrayBuffer>);
  mlp.forward(batchImages);
  const p = await mlp.P.toArray();
  const probs = p.slice(0, C);
  post({ type: 'probs', probs }, [probs.buffer]);
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
  }
};
