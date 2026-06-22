import { GpuTensor } from './tensor';
import matmulNaiveSrc from './shaders/matmul.wgsl?raw';
import matmulTiledSrc from './shaders/matmul_tiled.wgsl?raw';
import biasAddSrc from './shaders/bias_add.wgsl?raw';
import reluSrc from './shaders/relu.wgsl?raw';
import sigmoidSrc from './shaders/sigmoid.wgsl?raw';
import softmaxSrc from './shaders/softmax.wgsl?raw';
import crossEntropySrc from './shaders/cross_entropy.wgsl?raw';
import matmulATBSrc from './shaders/matmul_at_b.wgsl?raw';
import matmulABTSrc from './shaders/matmul_a_bt.wgsl?raw';
import biasBackwardSrc from './shaders/bias_backward.wgsl?raw';
import reluBackwardSrc from './shaders/relu_backward.wgsl?raw';
import sigmoidBackwardSrc from './shaders/sigmoid_backward.wgsl?raw';
import softmaxCeBackwardSrc from './shaders/softmax_ce_backward.wgsl?raw';
import sgdSrc from './shaders/sgd.wgsl?raw';
import adamSrc from './shaders/adam.wgsl?raw';
import gatherSrc from './shaders/gather.wgsl?raw';
import layerNormSrc from './shaders/layer_norm.wgsl?raw';
import causalSoftmaxSrc from './shaders/causal_softmax.wgsl?raw';
import geluSrc from './shaders/gelu.wgsl?raw';
import addSrc from './shaders/add.wgsl?raw';
import geluBackwardSrc from './shaders/gelu_backward.wgsl?raw';
import mulSrc from './shaders/mul.wgsl?raw';
import causalSoftmaxBackwardSrc from './shaders/causal_softmax_backward.wgsl?raw';
import layerNormBackwardSrc from './shaders/layer_norm_backward.wgsl?raw';
import sliceColsSrc from './shaders/slice_cols.wgsl?raw';
import pasteColsSrc from './shaders/paste_cols.wgsl?raw';

const WORKGROUP = 16;
const WG1D = 64;

interface DeviceCaches {
  modules: Map<string, GPUShaderModule>;
  pipelines: Map<string, GPUComputePipeline>;
}

const caches = new WeakMap<GPUDevice, DeviceCaches>();

function deviceCaches(device: GPUDevice): DeviceCaches {
  let c = caches.get(device);
  if (!c) {
    c = { modules: new Map(), pipelines: new Map() };
    caches.set(device, c);
  }
  return c;
}

function getModule(device: GPUDevice, key: string, code: string): GPUShaderModule {
  const { modules } = deviceCaches(device);
  let m = modules.get(key);
  if (!m) {
    m = device.createShaderModule({ label: key, code });
    void m.getCompilationInfo().then((info) => {
      for (const msg of info.messages) {
        if (msg.type === 'error') {
          console.error(`[WGSL ${key}] ${msg.lineNum}:${msg.linePos} — ${msg.message}`);
        }
      }
    });
    modules.set(key, m);
  }
  return m;
}

function getComputePipeline(
  device: GPUDevice,
  key: string,
  code: string,
  entryPoint = 'main',
): GPUComputePipeline {
  const { pipelines } = deviceCaches(device);
  let p = pipelines.get(key);
  if (!p) {
    p = device.createComputePipeline({
      label: key,
      layout: 'auto',
      compute: { module: getModule(device, key, code), entryPoint },
    });
    pipelines.set(key, p);
  }
  return p;
}

function metaBuffer(device: GPUDevice, values: number[]): GPUBuffer {
  const arr = new Uint32Array(4);
  for (let i = 0; i < values.length && i < 4; i++) arr[i] = values[i] >>> 0;
  const buf = device.createBuffer({
    label: 'meta',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arr);
  return buf;
}

function makeUniform(device: GPUDevice, byteLength: number, fill: (dv: DataView) => void): GPUBuffer {
  const ab = new ArrayBuffer(byteLength);
  fill(new DataView(ab));
  const buf = device.createBuffer({
    label: 'uniform',
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, new Uint8Array(ab));
  return buf;
}

export function createU32Buffer(device: GPUDevice, data: Uint32Array, label = 'u32'): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data as Uint32Array<ArrayBuffer>);
  return buf;
}

function dispatch1D(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  entries: GPUBindGroupEntry[],
  invocations: number,
  label: string,
): void {
  const bindGroup = device.createBindGroup({
    label: `${label}.bind`,
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(invocations / WG1D));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function dispatch2D(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  entries: GPUBindGroupEntry[],
  groupsX: number,
  groupsY: number,
  label: string,
): void {
  const bindGroup = device.createBindGroup({
    label: `${label}.bind`,
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

export type MatmulVariant = 'naive' | 'tiled';

export interface MatmulOptions {
  variant?: MatmulVariant;
  out?: GpuTensor;
}

export function matmul(
  device: GPUDevice,
  a: GpuTensor,
  b: GpuTensor,
  options: MatmulOptions = {},
): GpuTensor {
  if (a.shape.length !== 2 || b.shape.length !== 2) throw new Error('matmul expects 2-D tensors');
  const [M, K] = a.shape;
  const [K2, N] = b.shape;
  if (K !== K2) throw new Error(`matmul inner-dim mismatch: [${M}×${K}] @ [${K2}×${N}]`);

  const variant = options.variant ?? 'tiled';
  const out = options.out ?? GpuTensor.zeros(device, [M, N], { label: `matmul.out[${M}×${N}]` });
  if (out.shape[0] !== M || out.shape[1] !== N) {
    throw new Error(`matmul out shape [${out.shape.join('×')}] != expected [${M}×${N}]`);
  }

  const code = variant === 'tiled' ? matmulTiledSrc : matmulNaiveSrc;
  const pipeline = getComputePipeline(device, `matmul:${variant}`, code);
  dispatch2D(
    device,
    pipeline,
    [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: out.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [M, K, N]) } },
    ],
    Math.ceil(N / WORKGROUP),
    Math.ceil(M / WORKGROUP),
    `matmul:${variant}`,
  );
  return out;
}

function elementwise(
  device: GPUDevice,
  key: string,
  code: string,
  x: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  const o = out ?? GpuTensor.zeros(device, x.shape, { label: `${key}.out` });
  if (o.size !== x.size) throw new Error(`${key}: out size ${o.size} != input size ${x.size}`);
  dispatch1D(
    device,
    getComputePipeline(device, key, code),
    [
      { binding: 0, resource: { buffer: x.buffer } },
      { binding: 1, resource: { buffer: o.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [x.size]) } },
    ],
    x.size,
    key,
  );
  return o;
}

function elementwise2(
  device: GPUDevice,
  key: string,
  code: string,
  x0: GpuTensor,
  x1: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  if (x0.size !== x1.size) throw new Error(`${key}: input size mismatch ${x0.size} vs ${x1.size}`);
  const o = out ?? GpuTensor.zeros(device, x0.shape, { label: `${key}.out` });
  dispatch1D(
    device,
    getComputePipeline(device, key, code),
    [
      { binding: 0, resource: { buffer: x0.buffer } },
      { binding: 1, resource: { buffer: x1.buffer } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [x0.size]) } },
    ],
    x0.size,
    key,
  );
  return o;
}

export function relu(device: GPUDevice, x: GpuTensor, out?: GpuTensor): GpuTensor {
  return elementwise(device, 'relu', reluSrc, x, out);
}

export function sigmoid(device: GPUDevice, x: GpuTensor, out?: GpuTensor): GpuTensor {
  return elementwise(device, 'sigmoid', sigmoidSrc, x, out);
}

export function biasAdd(device: GPUDevice, y: GpuTensor, bias: GpuTensor): GpuTensor {
  if (y.shape.length !== 2) throw new Error('biasAdd: y must be 2-D');
  const [M, N] = y.shape;
  if (bias.shape.length !== 1 || bias.shape[0] !== N) {
    throw new Error(`biasAdd: bias must be [${N}], got [${bias.shape.join('×')}]`);
  }
  dispatch1D(
    device,
    getComputePipeline(device, 'bias_add', biasAddSrc),
    [
      { binding: 0, resource: { buffer: y.buffer } },
      { binding: 1, resource: { buffer: bias.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M * N,
    'bias_add',
  );
  return y;
}

export function softmax(device: GPUDevice, x: GpuTensor, out?: GpuTensor): GpuTensor {
  if (x.shape.length !== 2) throw new Error('softmax: input must be 2-D [rows, classes]');
  const [M, N] = x.shape;
  const o = out ?? GpuTensor.zeros(device, x.shape, { label: 'softmax.out' });
  dispatch1D(
    device,
    getComputePipeline(device, 'softmax', softmaxSrc),
    [
      { binding: 0, resource: { buffer: x.buffer } },
      { binding: 1, resource: { buffer: o.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M,
    'softmax',
  );
  return o;
}

export function crossEntropy(
  device: GPUDevice,
  probs: GpuTensor,
  labels: GPUBuffer,
  out?: GpuTensor,
): GpuTensor {
  if (probs.shape.length !== 2) throw new Error('crossEntropy: probs must be 2-D');
  const [M, N] = probs.shape;
  const losses = out ?? GpuTensor.zeros(device, [M], { label: 'ce.losses' });
  dispatch1D(
    device,
    getComputePipeline(device, 'cross_entropy', crossEntropySrc),
    [
      { binding: 0, resource: { buffer: probs.buffer } },
      { binding: 1, resource: { buffer: labels } },
      { binding: 2, resource: { buffer: losses.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M,
    'cross_entropy',
  );
  return losses;
}

export function matmulATB(device: GPUDevice, a: GpuTensor, b: GpuTensor, out?: GpuTensor): GpuTensor {
  const [M, K] = a.shape;
  const [M2, N] = b.shape;
  if (M !== M2) throw new Error(`matmulATB: row mismatch ${M} vs ${M2}`);
  const o = out ?? GpuTensor.zeros(device, [K, N], { label: 'matmulATB.out' });
  dispatch2D(
    device,
    getComputePipeline(device, 'matmul_at_b', matmulATBSrc),
    [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [M, K, N]) } },
    ],
    Math.ceil(N / WORKGROUP),
    Math.ceil(K / WORKGROUP),
    'matmul_at_b',
  );
  return o;
}

export function matmulABT(device: GPUDevice, a: GpuTensor, b: GpuTensor, out?: GpuTensor): GpuTensor {
  const [M, N] = a.shape;
  const [K, N2] = b.shape;
  if (N !== N2) throw new Error(`matmulABT: col mismatch ${N} vs ${N2}`);
  const o = out ?? GpuTensor.zeros(device, [M, K], { label: 'matmulABT.out' });
  dispatch2D(
    device,
    getComputePipeline(device, 'matmul_a_bt', matmulABTSrc),
    [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [M, N, K]) } },
    ],
    Math.ceil(K / WORKGROUP),
    Math.ceil(M / WORKGROUP),
    'matmul_a_bt',
  );
  return o;
}

export function biasBackward(device: GPUDevice, dY: GpuTensor, out?: GpuTensor): GpuTensor {
  if (dY.shape.length !== 2) throw new Error('biasBackward: dY must be 2-D');
  const [M, N] = dY.shape;
  const o = out ?? GpuTensor.zeros(device, [N], { label: 'db' });
  dispatch1D(
    device,
    getComputePipeline(device, 'bias_backward', biasBackwardSrc),
    [
      { binding: 0, resource: { buffer: dY.buffer } },
      { binding: 1, resource: { buffer: o.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    N,
    'bias_backward',
  );
  return o;
}

export function reluBackward(
  device: GPUDevice,
  dOut: GpuTensor,
  fwd: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  return elementwise2(device, 'relu_backward', reluBackwardSrc, dOut, fwd, out);
}

export function sigmoidBackward(
  device: GPUDevice,
  dOut: GpuTensor,
  y: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  return elementwise2(device, 'sigmoid_backward', sigmoidBackwardSrc, dOut, y, out);
}

export function softmaxCeBackward(
  device: GPUDevice,
  probs: GpuTensor,
  labels: GPUBuffer,
  out?: GpuTensor,
): GpuTensor {
  if (probs.shape.length !== 2) throw new Error('softmaxCeBackward: probs must be 2-D');
  const [M, N] = probs.shape;
  const o = out ?? GpuTensor.zeros(device, [M, N], { label: 'dLogits' });
  dispatch1D(
    device,
    getComputePipeline(device, 'softmax_ce_backward', softmaxCeBackwardSrc),
    [
      { binding: 0, resource: { buffer: probs.buffer } },
      { binding: 1, resource: { buffer: labels } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M * N,
    'softmax_ce_backward',
  );
  return o;
}

export function sgdStep(device: GPUDevice, w: GpuTensor, g: GpuTensor, lr: number): void {
  if (w.size !== g.size) throw new Error(`sgdStep: size mismatch ${w.size} vs ${g.size}`);
  const u = makeUniform(device, 16, (dv) => {
    dv.setUint32(0, w.size, true);
    dv.setFloat32(4, lr, true);
  });
  dispatch1D(
    device,
    getComputePipeline(device, 'sgd', sgdSrc),
    [
      { binding: 0, resource: { buffer: w.buffer } },
      { binding: 1, resource: { buffer: g.buffer } },
      { binding: 2, resource: { buffer: u } },
    ],
    w.size,
    'sgd',
  );
}

export interface AdamStepParams {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  bc1: number;
  bc2: number;
}

export function adamStep(
  device: GPUDevice,
  w: GpuTensor,
  g: GpuTensor,
  m: GpuTensor,
  v: GpuTensor,
  p: AdamStepParams,
): void {
  const u = makeUniform(device, 32, (dv) => {
    dv.setUint32(0, w.size, true);
    dv.setFloat32(4, p.lr, true);
    dv.setFloat32(8, p.beta1, true);
    dv.setFloat32(12, p.beta2, true);
    dv.setFloat32(16, p.eps, true);
    dv.setFloat32(20, p.bc1, true);
    dv.setFloat32(24, p.bc2, true);
  });
  dispatch1D(
    device,
    getComputePipeline(device, 'adam', adamSrc),
    [
      { binding: 0, resource: { buffer: w.buffer } },
      { binding: 1, resource: { buffer: g.buffer } },
      { binding: 2, resource: { buffer: m.buffer } },
      { binding: 3, resource: { buffer: v.buffer } },
      { binding: 4, resource: { buffer: u } },
    ],
    w.size,
    'adam',
  );
}

export function gather(
  device: GPUDevice,
  packed: GPUBuffer,
  indices: GPUBuffer,
  batchSize: number,
  pixels: number,
  out?: GpuTensor,
): GpuTensor {
  const o = out ?? GpuTensor.zeros(device, [batchSize, pixels], { label: 'batch' });
  dispatch1D(
    device,
    getComputePipeline(device, 'gather', gatherSrc),
    [
      { binding: 0, resource: { buffer: packed } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: metaBuffer(device, [batchSize, pixels]) } },
    ],
    batchSize * pixels,
    'gather',
  );
  return o;
}

export function gelu(device: GPUDevice, x: GpuTensor, out?: GpuTensor): GpuTensor {
  return elementwise(device, 'gelu', geluSrc, x, out);
}

export function addTensors(
  device: GPUDevice,
  a: GpuTensor,
  b: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  return elementwise2(device, 'add', addSrc, a, b, out);
}

export function layerNorm(
  device: GPUDevice,
  x: GpuTensor,
  gamma: GpuTensor,
  beta: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  if (x.shape.length !== 2) throw new Error('layerNorm: x must be 2-D');
  const [M, N] = x.shape;
  if (gamma.size !== N || beta.size !== N) throw new Error(`layerNorm: γ/β must be [${N}]`);
  const o = out ?? GpuTensor.zeros(device, x.shape, { label: 'layer_norm.out' });
  dispatch1D(
    device,
    getComputePipeline(device, 'layer_norm', layerNormSrc),
    [
      { binding: 0, resource: { buffer: x.buffer } },
      { binding: 1, resource: { buffer: gamma.buffer } },
      { binding: 2, resource: { buffer: beta.buffer } },
      { binding: 3, resource: { buffer: o.buffer } },
      { binding: 4, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M,
    'layer_norm',
  );
  return o;
}

export function causalSoftmax(
  device: GPUDevice,
  scores: GpuTensor,
  scale: number,
  out?: GpuTensor,
): GpuTensor {
  if (scores.shape.length !== 2 || scores.shape[0] !== scores.shape[1]) {
    throw new Error('causalSoftmax: scores must be square [T×T]');
  }
  const T = scores.shape[0];
  const o = out ?? GpuTensor.zeros(device, scores.shape, { label: 'causal_softmax.out' });
  const u = makeUniform(device, 16, (dv) => {
    dv.setUint32(0, T, true);
    dv.setFloat32(4, scale, true);
  });
  dispatch1D(
    device,
    getComputePipeline(device, 'causal_softmax', causalSoftmaxSrc),
    [
      { binding: 0, resource: { buffer: scores.buffer } },
      { binding: 1, resource: { buffer: o.buffer } },
      { binding: 2, resource: { buffer: u } },
    ],
    T,
    'causal_softmax',
  );
  return o;
}

export function geluBackward(
  device: GPUDevice,
  dOut: GpuTensor,
  pre: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  return elementwise2(device, 'gelu_backward', geluBackwardSrc, dOut, pre, out);
}

export function mulTensors(
  device: GPUDevice,
  a: GpuTensor,
  b: GpuTensor,
  out?: GpuTensor,
): GpuTensor {
  return elementwise2(device, 'mul', mulSrc, a, b, out);
}

export function causalSoftmaxBackward(
  device: GPUDevice,
  attn: GpuTensor,
  dattn: GpuTensor,
  scale: number,
  out?: GpuTensor,
): GpuTensor {
  if (attn.shape.length !== 2 || attn.shape[0] !== attn.shape[1]) {
    throw new Error('causalSoftmaxBackward: attn must be square [T×T]');
  }
  const T = attn.shape[0];
  const o = out ?? GpuTensor.zeros(device, attn.shape, { label: 'causal_softmax_backward.out' });
  const u = makeUniform(device, 16, (dv) => {
    dv.setUint32(0, T, true);
    dv.setFloat32(4, scale, true);
  });
  dispatch1D(
    device,
    getComputePipeline(device, 'causal_softmax_backward', causalSoftmaxBackwardSrc),
    [
      { binding: 0, resource: { buffer: attn.buffer } },
      { binding: 1, resource: { buffer: dattn.buffer } },
      { binding: 2, resource: { buffer: o.buffer } },
      { binding: 3, resource: { buffer: u } },
    ],
    T,
    'causal_softmax_backward',
  );
  return o;
}

export function layerNormBackward(
  device: GPUDevice,
  x: GpuTensor,
  dy: GpuTensor,
  gamma: GpuTensor,
): { dx: GpuTensor; dgamma: GpuTensor; dbeta: GpuTensor } {
  if (x.shape.length !== 2) throw new Error('layerNormBackward: x must be 2-D');
  const [M, N] = x.shape;
  const dx = GpuTensor.zeros(device, x.shape, { label: 'ln_bwd.dx' });
  const xhat = GpuTensor.zeros(device, x.shape, { label: 'ln_bwd.xhat' });
  dispatch1D(
    device,
    getComputePipeline(device, 'layer_norm_backward', layerNormBackwardSrc),
    [
      { binding: 0, resource: { buffer: x.buffer } },
      { binding: 1, resource: { buffer: dy.buffer } },
      { binding: 2, resource: { buffer: gamma.buffer } },
      { binding: 3, resource: { buffer: dx.buffer } },
      { binding: 4, resource: { buffer: xhat.buffer } },
      { binding: 5, resource: { buffer: metaBuffer(device, [M, N]) } },
    ],
    M,
    'layer_norm_backward',
  );
  const dbeta = biasBackward(device, dy);
  const tmp = mulTensors(device, dy, xhat);
  const dgamma = biasBackward(device, tmp);
  tmp.destroy();
  xhat.destroy();
  return { dx, dgamma, dbeta };
}

export function sliceCols(
  device: GPUDevice,
  src: GpuTensor,
  c0: number,
  w: number,
  out?: GpuTensor,
): GpuTensor {
  if (src.shape.length !== 2) throw new Error('sliceCols: src must be 2-D');
  const [M, N] = src.shape;
  const o = out ?? GpuTensor.zeros(device, [M, w], { label: 'slice_cols.out' });
  dispatch1D(
    device,
    getComputePipeline(device, 'slice_cols', sliceColsSrc),
    [
      { binding: 0, resource: { buffer: src.buffer } },
      { binding: 1, resource: { buffer: o.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [M, N, c0, w]) } },
    ],
    M * w,
    'slice_cols',
  );
  return o;
}

export function pasteCols(device: GPUDevice, dst: GpuTensor, src: GpuTensor, c0: number): GpuTensor {
  if (dst.shape.length !== 2 || src.shape.length !== 2) throw new Error('pasteCols: 2-D only');
  const [M, N] = dst.shape;
  const w = src.shape[1];
  dispatch1D(
    device,
    getComputePipeline(device, 'paste_cols', pasteColsSrc),
    [
      { binding: 0, resource: { buffer: src.buffer } },
      { binding: 1, resource: { buffer: dst.buffer } },
      { binding: 2, resource: { buffer: metaBuffer(device, [M, N, c0, w]) } },
    ],
    M * w,
    'paste_cols',
  );
  return dst;
}
