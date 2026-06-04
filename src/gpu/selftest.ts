// Self-verification harness. Every GPU kernel is run against the f64 CPU oracle
// across a spread of shapes (including non-multiples of the tile size, to
// exercise boundary masking). The backward pass is additionally gated by
// numerical gradient checking. Runs on app load so correctness is proven in the
// user's real browser, not just asserted.

import { getGpuContext } from './device';
import { GpuTensor } from './tensor';
import {
  matmul,
  relu,
  sigmoid,
  biasAdd,
  softmax,
  crossEntropy,
  matmulATB,
  matmulABT,
  biasBackward,
  reluBackward,
  sigmoidBackward,
  softmaxCeBackward,
  gather,
  createU32Buffer,
  type MatmulVariant,
} from './ops';
import * as ref from '../nn/reference';
import { Mlp } from '../nn/mlp';
import { numericalGradCheck } from '../nn/gradcheck';
import { SGD, Adam } from '../nn/optimizer';
import { makeBlobs, gaussian } from '../data/synthetic';

const TOL = 1e-3; // generous headroom for f32 vs f64 accumulation
const GRAD_TOL = 2e-2; // central-difference + f32 readback noise

export type CheckGroup =
  | 'matmul'
  | 'forward'
  | 'composed'
  | 'backward'
  | 'gradcheck'
  | 'optim'
  | 'data';

export interface CheckResult {
  group: CheckGroup;
  name: string;
  detail: string;
  maxAbsErr: number;
  maxRelErr: number;
  ms: number;
  pass: boolean;
  throughput?: string;
}

/** Result of the synthetic training demo — the "it actually learns" proof. */
export interface TrainingResult {
  arch: string;
  optimizer: string;
  points: number;
  steps: number;
  initialLoss: number;
  finalLoss: number;
  accuracy: number;
  lossHistory: number[];
  ms: number;
  pass: boolean;
}

export interface SelfTestReport {
  results: CheckResult[];
  training: TrainingResult | null;
  allPassed: boolean;
}

// Deterministic PRNG (mulberry32) so reported errors are reproducible.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randRange(n: number, range: number, rng: () => number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rng() * 2 - 1) * range;
  return out;
}

function verdict(
  group: CheckGroup,
  name: string,
  detail: string,
  m: ref.ErrorMetrics,
  ms: number,
  tol = TOL,
): CheckResult {
  return {
    group,
    name,
    detail,
    maxAbsErr: m.maxAbsErr,
    maxRelErr: m.maxRelErr,
    ms,
    pass: Number.isFinite(m.maxRelErr) && m.maxRelErr < tol,
  };
}

// ---- matmul ----

async function matmulCheck(M: number, K: number, N: number, variant: MatmulVariant): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(1234);
  const aData = randRange(M * K, 1, rng);
  const bData = randRange(K * N, 1, rng);
  const a = GpuTensor.fromArray(device, aData, [M, K]);
  const b = GpuTensor.fromArray(device, bData, [K, N]);

  const t0 = performance.now();
  const c = matmul(device, a, b, { variant });
  const got = await c.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.matmulCpu(aData, bData, M, K, N));
  a.destroy();
  b.destroy();
  c.destroy();

  const res = verdict('matmul', variant, `${M}×${K} · ${K}×${N}`, m, ms);
  const gflops = (2 * M * K * N) / (ms / 1000) / 1e9;
  if (gflops >= 1) res.throughput = `${gflops.toFixed(1)} GF/s`;
  return res;
}

// ---- forward: elementwise activations ----

async function activationCheck(
  name: string,
  gpuFn: (d: GPUDevice, x: GpuTensor) => GpuTensor,
  cpuFn: (x: Float32Array) => Float32Array,
  M = 64,
  N = 64,
  range = 4,
  seed = 7,
): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const x = randRange(M * N, range, mulberry32(seed));
  const tx = GpuTensor.fromArray(device, x, [M, N]);

  const t0 = performance.now();
  const ty = gpuFn(device, tx);
  const got = await ty.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, cpuFn(x));
  tx.destroy();
  ty.destroy();
  return verdict('forward', name, `${M}×${N} elementwise · in ∈ [-${range}, ${range}]`, m, ms);
}

async function biasAddCheck(M = 32, N = 48, seed = 11): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const yData = randRange(M * N, 2, rng);
  const bData = randRange(N, 2, rng);
  const ty = GpuTensor.fromArray(device, yData, [M, N]);
  const tb = GpuTensor.fromArray(device, bData, [N]);

  const t0 = performance.now();
  biasAdd(device, ty, tb);
  const got = await ty.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.biasAddCpu(yData, bData, M, N));
  ty.destroy();
  tb.destroy();
  return verdict('forward', 'bias add', `[${M}×${N}] + [${N}] broadcast`, m, ms);
}

async function softmaxCheck(M = 64, N = 10, seed = 13): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const x = randRange(M * N, 4, mulberry32(seed));
  const tx = GpuTensor.fromArray(device, x, [M, N]);

  const t0 = performance.now();
  const ty = softmax(device, tx);
  const got = await ty.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.softmaxCpu(x, M, N));
  tx.destroy();
  ty.destroy();
  return verdict('forward', 'softmax', `${M} rows × ${N} classes · max-subtraction`, m, ms);
}

async function crossEntropyCheck(M = 64, N = 10, seed = 17): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const logits = randRange(M * N, 4, rng);
  const labels = new Uint32Array(M);
  for (let i = 0; i < M; i++) labels[i] = Math.floor(rng() * N);

  const probsCpu = ref.softmaxCpu(logits, M, N);
  const tx = GpuTensor.fromArray(device, logits, [M, N]);
  const tp = softmax(device, tx);
  const labelBuf = createU32Buffer(device, labels, 'labels');

  const t0 = performance.now();
  const tl = crossEntropy(device, tp, labelBuf);
  const got = await tl.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.crossEntropyCpu(probsCpu, labels, M, N));
  tx.destroy();
  tp.destroy();
  tl.destroy();
  return verdict('forward', 'cross-entropy', `${M} samples · ${N} classes`, m, ms);
}

// ---- composed forward (full MLP) ----

async function mlpForwardCheck(seed = 23): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const B = 8, D0 = 16, D1 = 12, C = 10;
  const rng = mulberry32(seed);

  const X = randRange(B * D0, 1, rng);
  const W1 = randRange(D0 * D1, 0.5, rng);
  const b1 = randRange(D1, 0.1, rng);
  const W2 = randRange(D1 * C, 0.5, rng);
  const b2 = randRange(C, 0.1, rng);
  const labels = new Uint32Array(B);
  for (let i = 0; i < B; i++) labels[i] = Math.floor(rng() * C);

  const tX = GpuTensor.fromArray(device, X, [B, D0]);
  const tW1 = GpuTensor.fromArray(device, W1, [D0, D1]);
  const tb1 = GpuTensor.fromArray(device, b1, [D1]);
  const tW2 = GpuTensor.fromArray(device, W2, [D1, C]);
  const tb2 = GpuTensor.fromArray(device, b2, [C]);
  const labelBuf = createU32Buffer(device, labels, 'labels');

  const t0 = performance.now();
  const tZ1 = matmul(device, tX, tW1);
  biasAdd(device, tZ1, tb1);
  const tA1 = relu(device, tZ1);
  const tZ2 = matmul(device, tA1, tW2);
  biasAdd(device, tZ2, tb2);
  const tP = softmax(device, tZ2);
  const tL = crossEntropy(device, tP, labelBuf);
  const gotP = await tP.toArray();
  const gotL = await tL.toArray();
  const ms = performance.now() - t0;

  const Z1 = ref.biasAddCpu(ref.matmulCpu(X, W1, B, D0, D1), b1, B, D1);
  const A1 = ref.reluCpu(Z1);
  const Z2 = ref.biasAddCpu(ref.matmulCpu(A1, W2, B, D1, C), b2, B, C);
  const P = ref.softmaxCpu(Z2, B, C);
  const L = ref.crossEntropyCpu(P, labels, B, C);

  const mp = ref.compareArrays(gotP, P);
  const ml = ref.compareArrays(gotL, L);
  const combined: ref.ErrorMetrics = {
    maxAbsErr: Math.max(mp.maxAbsErr, ml.maxAbsErr),
    maxRelErr: Math.max(mp.maxRelErr, ml.maxRelErr),
  };

  for (const t of [tX, tW1, tb1, tW2, tb2, tZ1, tA1, tZ2, tP, tL]) t.destroy();

  const meanLoss = L.reduce((a, b) => a + b, 0) / B;
  return verdict(
    'composed',
    'MLP forward',
    `${D0}→${D1} (relu) →${C} · softmax+CE · loss ≈ ${meanLoss.toFixed(3)}`,
    combined,
    ms,
    TOL * 2,
  );
}

// ---- backward ops (GPU vs CPU oracle) ----

async function matmulATBCheck(M = 64, K = 32, N = 48, seed = 41): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const a = randRange(M * K, 1, rng);
  const b = randRange(M * N, 1, rng);
  const ta = GpuTensor.fromArray(device, a, [M, K]);
  const tb = GpuTensor.fromArray(device, b, [M, N]);

  const t0 = performance.now();
  const tc = matmulATB(device, ta, tb);
  const got = await tc.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.matmulATBCpu(a, b, M, K, N));
  ta.destroy();
  tb.destroy();
  tc.destroy();
  return verdict('backward', 'matmul Aᵀ·B', `dW = Xᵀ@dY → [${K}×${N}]`, m, ms);
}

async function matmulABTCheck(M = 64, N = 48, K = 32, seed = 43): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const a = randRange(M * N, 1, rng);
  const b = randRange(K * N, 1, rng);
  const ta = GpuTensor.fromArray(device, a, [M, N]);
  const tb = GpuTensor.fromArray(device, b, [K, N]);

  const t0 = performance.now();
  const tc = matmulABT(device, ta, tb);
  const got = await tc.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.matmulABTCpu(a, b, M, N, K));
  ta.destroy();
  tb.destroy();
  tc.destroy();
  return verdict('backward', 'matmul A·Bᵀ', `dX = dY@Wᵀ → [${M}×${K}]`, m, ms);
}

async function biasBackwardCheck(M = 64, N = 10, seed = 47): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const dY = randRange(M * N, 1, mulberry32(seed));
  const tdY = GpuTensor.fromArray(device, dY, [M, N]);

  const t0 = performance.now();
  const tdb = biasBackward(device, tdY);
  const got = await tdb.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.biasBackwardCpu(dY, M, N));
  tdY.destroy();
  tdb.destroy();
  return verdict('backward', 'bias backward', `Σ over ${M} rows → [${N}]`, m, ms);
}

async function reluBackwardCheck(M = 64, N = 64, seed = 53): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const dOut = randRange(M * N, 1, rng);
  const fwd = randRange(M * N, 4, rng);
  const tdOut = GpuTensor.fromArray(device, dOut, [M, N]);
  const tfwd = GpuTensor.fromArray(device, fwd, [M, N]);

  const t0 = performance.now();
  const tdx = reluBackward(device, tdOut, tfwd);
  const got = await tdx.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.reluBackwardCpu(dOut, fwd));
  tdOut.destroy();
  tfwd.destroy();
  tdx.destroy();
  return verdict('backward', "ReLU′", `${M}×${N} · dOut ⊙ (x>0)`, m, ms);
}

async function sigmoidBackwardCheck(M = 64, N = 64, seed = 59): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const dOut = randRange(M * N, 1, rng);
  const y = ref.sigmoidCpu(randRange(M * N, 3, rng)); // valid y ∈ (0,1)
  const tdOut = GpuTensor.fromArray(device, dOut, [M, N]);
  const ty = GpuTensor.fromArray(device, y, [M, N]);

  const t0 = performance.now();
  const tdx = sigmoidBackward(device, tdOut, ty);
  const got = await tdx.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.sigmoidBackwardCpu(dOut, y));
  tdOut.destroy();
  ty.destroy();
  tdx.destroy();
  return verdict('backward', "sigmoid′", `${M}×${N} · dOut·y·(1−y)`, m, ms);
}

async function softmaxCeBackwardCheck(M = 64, N = 10, seed = 61): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const logits = randRange(M * N, 4, rng);
  const labels = new Uint32Array(M);
  for (let i = 0; i < M; i++) labels[i] = Math.floor(rng() * N);

  const probsCpu = ref.softmaxCpu(logits, M, N);
  const tx = GpuTensor.fromArray(device, logits, [M, N]);
  const tp = softmax(device, tx);
  const labelBuf = createU32Buffer(device, labels, 'labels');

  const t0 = performance.now();
  const td = softmaxCeBackward(device, tp, labelBuf);
  const got = await td.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.softmaxCeBackwardCpu(probsCpu, labels, M, N));
  tx.destroy();
  tp.destroy();
  td.destroy();
  return verdict('backward', 'softmax+CE backward', `(p − onehot)/M → [${M}×${N}]`, m, ms);
}

// ---- gradient check (numerical vs analytic, end-to-end) ----

async function gradCheckSuite(seed = 31): Promise<CheckResult[]> {
  const { device } = await getGpuContext();
  const B = 4, D0 = 5, D1 = 4, C = 3;
  const rng = mulberry32(seed);

  const init = {
    W1: randRange(D0 * D1, 0.6, rng),
    b1: randRange(D1, 0.2, rng),
    W2: randRange(D1 * C, 0.6, rng),
    b2: randRange(C, 0.2, rng),
  };
  const Xd = randRange(B * D0, 1, rng);
  const labels = new Uint32Array(B);
  for (let i = 0; i < B; i++) labels[i] = Math.floor(rng() * C);

  const mlp = new Mlp(device, { inputDim: D0, hiddenDim: D1, outputDim: C, batch: B }, init);
  const X = GpuTensor.fromArray(device, Xd, [B, D0], { label: 'X' });
  const labelBuf = createU32Buffer(device, labels, 'labels');

  // analytic gradients from one forward + backward
  await mlp.forwardLoss(X, labelBuf);
  mlp.backward(labelBuf);

  const targets: Array<[string, GpuTensor, GpuTensor, string]> = [
    ['∂L/∂W1', mlp.W1, mlp.dW1, `[${D0}×${D1}]`],
    ['∂L/∂b1', mlp.b1, mlp.db1, `[${D1}]`],
    ['∂L/∂W2', mlp.W2, mlp.dW2, `[${D1}×${C}]`],
    ['∂L/∂b2', mlp.b2, mlp.db2, `[${C}]`],
    ['∂L/∂X', X, mlp.dX, `[${B}×${D0}]`],
  ];

  const results: CheckResult[] = [];
  for (const [name, param, grad, shape] of targets) {
    const t0 = performance.now();
    const m = await numericalGradCheck(device, mlp, X, labelBuf, param, grad);
    const ms = performance.now() - t0;
    results.push({
      group: 'gradcheck',
      name,
      detail: `${shape} · central diff ε=1e-2`,
      maxAbsErr: m.maxAbsErr,
      maxRelErr: m.maxRelErr,
      ms,
      pass: Number.isFinite(m.maxRelErr) && m.maxRelErr < GRAD_TOL,
    });
  }

  mlp.destroy();
  X.destroy();
  return results;
}

// ---- optimizers (GPU vs CPU oracle) ----

async function sgdCheck(n = 256, seed = 67): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const w0 = randRange(n, 1, rng);
  const g = randRange(n, 0.5, rng);
  const lr = 0.1;
  const tw = GpuTensor.fromArray(device, w0, [n], { label: 'w' });
  const tg = GpuTensor.fromArray(device, g, [n], { label: 'g' });
  const opt = new SGD(device, [tw], lr);

  const t0 = performance.now();
  opt.step([tg]);
  const got = await tw.toArray();
  const ms = performance.now() - t0;

  const m = ref.compareArrays(got, ref.sgdStepCpu(w0, g, lr));
  tw.destroy();
  tg.destroy();
  return verdict('optim', 'SGD step', `${n} params · w −= lr·g`, m, ms);
}

async function adamCheck(n = 256, steps = 3, seed = 71): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const w0 = randRange(n, 1, rng);
  const g = randRange(n, 0.5, rng);
  const cfg = { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
  const tw = GpuTensor.fromArray(device, w0, [n], { label: 'w' });
  const tg = GpuTensor.fromArray(device, g, [n], { label: 'g' });
  const opt = new Adam(device, [tw], cfg);

  const t0 = performance.now();
  for (let s = 0; s < steps; s++) opt.step([tg]);
  const got = await tw.toArray();
  const ms = performance.now() - t0;

  const grads = Array.from({ length: steps }, () => g);
  const m = ref.compareArrays(got, ref.adamCpu(w0, grads, cfg));
  tw.destroy();
  tg.destroy();
  opt.destroy();
  return verdict('optim', `Adam ×${steps}`, `${n} params · m,v + bias-correction`, m, ms);
}

// ---- training demo (synthetic data, full-batch) ----

async function trainingDemo(): Promise<TrainingResult> {
  const { device } = await getGpuContext();
  const D0 = 16, D1 = 32, C = 4, N = 256, STEPS = 120;
  const ds = makeBlobs(N, D0, C, { seed: 2, spread: 0.7, separation: 3 });

  // He init for the ReLU layer; zero biases.
  const rng = mulberry32(99);
  const heInit = (fanIn: number, len: number): Float32Array => {
    const std = Math.sqrt(2 / fanIn);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = gaussian(rng) * std;
    return out;
  };
  const init = {
    W1: heInit(D0, D0 * D1),
    b1: new Float32Array(D1),
    W2: heInit(D1, D1 * C),
    b2: new Float32Array(C),
  };

  const mlp = new Mlp(device, { inputDim: D0, hiddenDim: D1, outputDim: C, batch: N }, init);
  const X = GpuTensor.fromArray(device, ds.X, [N, D0], { label: 'X' });
  const labelBuf = createU32Buffer(device, ds.labels, 'labels');
  const opt = new Adam(device, mlp.params(), { lr: 0.02 });

  const t0 = performance.now();
  let loss = await mlp.forwardLoss(X, labelBuf); // caches P at the initial params
  const initialLoss = loss;
  const lossHistory: number[] = [loss];
  for (let s = 0; s < STEPS; s++) {
    mlp.backward(labelBuf); // uses the P cached by the most recent forwardLoss
    opt.step(mlp.grads());
    loss = await mlp.forwardLoss(X, labelBuf); // forward at updated params
    lossHistory.push(loss);
  }
  // accuracy from the final cached probabilities
  const P = await mlp.P.toArray();
  const acc = ref.accuracy(P, ds.labels, N, C);
  const ms = performance.now() - t0;

  mlp.destroy();
  X.destroy();
  opt.destroy();

  return {
    arch: `${D0}→${D1} (relu) →${C}`,
    optimizer: 'Adam · lr 0.02',
    points: N,
    steps: STEPS,
    initialLoss,
    finalLoss: loss,
    accuracy: acc,
    lossHistory,
    ms,
    pass: loss < 0.3 && acc > 0.9,
  };
}

// ---- data pipeline (gather, synthetic — no network) ----

async function gatherCheck(images = 20, pixels = 16, batch = 5, seed = 83): Promise<CheckResult> {
  const { device } = await getGpuContext();
  const rng = mulberry32(seed);
  const bytes = new Uint8Array(images * pixels);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(rng() * 256);
  const packed = device.createBuffer({
    label: 'gatherCheck.packed',
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(packed, 0, bytes);

  const idx = new Uint32Array(batch);
  for (let i = 0; i < batch; i++) idx[i] = Math.floor(rng() * images);
  const idxBuf = createU32Buffer(device, idx, 'gatherCheck.idx');

  const t0 = performance.now();
  const out = gather(device, packed, idxBuf, batch, pixels);
  const got = await out.toArray();
  const ms = performance.now() - t0;

  const refArr = new Float32Array(batch * pixels);
  for (let b = 0; b < batch; b++) {
    for (let p = 0; p < pixels; p++) refArr[b * pixels + p] = bytes[idx[b] * pixels + p] / 255;
  }
  const m = ref.compareArrays(got, refArr);
  packed.destroy();
  out.destroy();
  return verdict('data', 'gather batch', `${batch}×${pixels} from ${images} imgs · uint8→f32/255`, m, ms);
}

// ---- runner ----

const MATMUL_SHAPES: Array<[number, number, number]> = [
  [1, 1, 1],
  [2, 3, 4],
  [17, 33, 19], // non-multiples of TILE — exercises boundary masking
  [64, 128, 32],
  [128, 256, 128],
  [256, 256, 256],
];

export async function runSelfTest(): Promise<SelfTestReport> {
  const results: CheckResult[] = [];

  for (const [M, K, N] of MATMUL_SHAPES) {
    for (const variant of ['naive', 'tiled'] as const) {
      results.push(await matmulCheck(M, K, N, variant));
    }
  }

  results.push(await activationCheck('relu', relu, ref.reluCpu));
  results.push(await activationCheck('sigmoid', sigmoid, ref.sigmoidCpu, 64, 64, 8));
  results.push(await biasAddCheck());
  results.push(await softmaxCheck());
  results.push(await crossEntropyCheck());

  results.push(await mlpForwardCheck());

  results.push(await matmulATBCheck());
  results.push(await matmulABTCheck());
  results.push(await biasBackwardCheck());
  results.push(await reluBackwardCheck());
  results.push(await sigmoidBackwardCheck());
  results.push(await softmaxCeBackwardCheck());

  results.push(...(await gradCheckSuite()));

  results.push(await sgdCheck());
  results.push(await adamCheck());

  results.push(await gatherCheck());

  const training = await trainingDemo();

  const allPassed = results.every((r) => r.pass) && training.pass;
  return { results, training, allPassed };
}
