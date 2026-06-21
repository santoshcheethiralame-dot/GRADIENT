import { GpuTensor } from '../gpu/tensor';
import {
  matmul,
  matmulATB,
  matmulABT,
  biasAdd,
  biasBackward,
  softmax,
  softmaxCeBackward,
  layerNorm,
  layerNormBackward,
  causalSoftmax,
  causalSoftmaxBackward,
  gelu,
  geluBackward,
  addTensors,
  createU32Buffer,
} from '../gpu/ops';
import type { NanoGpt } from './nanogpt';

export async function nanoGptGpuForward(
  device: GPUDevice,
  model: NanoGpt,
  ids: number[],
): Promise<Float32Array> {
  const { dEmbed: dE, vocab } = model.cfg;
  const t = ids.length;

  const xData = new Float32Array(t * dE);
  for (let i = 0; i < t; i++) {
    const te = ids[i] * dE;
    const pe = i * dE;
    for (let c = 0; c < dE; c++) xData[i * dE + c] = model.tokEmb[te + c] + model.posEmb[pe + c];
  }

  const tensors: GpuTensor[] = [];
  const keep = <T extends GpuTensor>(g: T): T => {
    tensors.push(g);
    return g;
  };
  const up = (arr: Float32Array, shape: number[]): GpuTensor =>
    keep(GpuTensor.fromArray(device, arr, shape));

  const x = up(xData, [t, dE]);
  const ln1g = up(model.ln1g, [dE]);
  const ln1b = up(model.ln1b, [dE]);
  const Wq = up(model.Wq, [dE, dE]);
  const Wk = up(model.Wk, [dE, dE]);
  const Wv = up(model.Wv, [dE, dE]);
  const Wo = up(model.Wo, [dE, dE]);
  const ln2g = up(model.ln2g, [dE]);
  const ln2b = up(model.ln2b, [dE]);
  const Wff1 = up(model.Wff1, [dE, model.cfg.dFF]);
  const bff1 = up(model.bff1, [model.cfg.dFF]);
  const Wff2 = up(model.Wff2, [model.cfg.dFF, dE]);
  const bff2 = up(model.bff2, [dE]);
  const lnfg = up(model.lnfg, [dE]);
  const lnfb = up(model.lnfb, [dE]);
  const head = up(model.head, [dE, vocab]);

  const h = keep(layerNorm(device, x, ln1g, ln1b));
  const Q = keep(matmul(device, h, Wq));
  const K = keep(matmul(device, h, Wk));
  const V = keep(matmul(device, h, Wv));
  const scores = keep(matmulABT(device, Q, K));
  const attn = keep(causalSoftmax(device, scores, 1 / Math.sqrt(dE)));
  const ctx = keep(matmul(device, attn, V));
  const o = keep(matmul(device, ctx, Wo));
  const xa = keep(addTensors(device, x, o));

  const h2 = keep(layerNorm(device, xa, ln2g, ln2b));
  const f = keep(matmul(device, h2, Wff1));
  biasAdd(device, f, bff1);
  const fg = keep(gelu(device, f));
  const f2 = keep(matmul(device, fg, Wff2));
  biasAdd(device, f2, bff2);
  const xb = keep(addTensors(device, xa, f2));

  const xf = keep(layerNorm(device, xb, lnfg, lnfb));
  const logits = keep(matmul(device, xf, head));

  const result = await logits.toArray();
  for (const tn of tensors) tn.destroy();
  return result;
}

// The full backward pass on the GPU. Recomputes the forward (retaining every
// intermediate), backprops with the same math as NanoGpt.backward, and returns
// each parameter gradient as a Float32Array keyed by the params() name — so it
// can be diffed directly against the CPU backward. The embedding lookup and its
// scatter-add gradient are done on the CPU (a tiny gather); everything else runs
// on the GPU and reuses matmul / matmulATB / matmulABT plus the new kernels.
export async function nanoGptGpuBackward(
  device: GPUDevice,
  model: NanoGpt,
  ids: number[],
  targets: number[],
): Promise<Record<string, Float32Array>> {
  const { dEmbed: dE, dFF, vocab } = model.cfg;
  const t = ids.length;
  const scale = 1 / Math.sqrt(dE);

  const tensors: GpuTensor[] = [];
  const keep = <T extends GpuTensor>(g: T): T => {
    tensors.push(g);
    return g;
  };
  const up = (arr: Float32Array, shape: number[]): GpuTensor =>
    keep(GpuTensor.fromArray(device, arr, shape));

  const xData = new Float32Array(t * dE);
  for (let i = 0; i < t; i++) {
    const te = ids[i] * dE;
    const pe = i * dE;
    for (let c = 0; c < dE; c++) xData[i * dE + c] = model.tokEmb[te + c] + model.posEmb[pe + c];
  }
  const x = up(xData, [t, dE]);
  const ln1g = up(model.ln1g, [dE]);
  const ln1b = up(model.ln1b, [dE]);
  const Wq = up(model.Wq, [dE, dE]);
  const Wk = up(model.Wk, [dE, dE]);
  const Wv = up(model.Wv, [dE, dE]);
  const Wo = up(model.Wo, [dE, dE]);
  const ln2g = up(model.ln2g, [dE]);
  const ln2b = up(model.ln2b, [dE]);
  const Wff1 = up(model.Wff1, [dE, dFF]);
  const bff1 = up(model.bff1, [dFF]);
  const Wff2 = up(model.Wff2, [dFF, dE]);
  const bff2 = up(model.bff2, [dE]);
  const lnfg = up(model.lnfg, [dE]);
  const lnfb = up(model.lnfb, [dE]);
  const head = up(model.head, [dE, vocab]);

  // ---- forward (retain intermediates) ----
  const h = keep(layerNorm(device, x, ln1g, ln1b));
  const Q = keep(matmul(device, h, Wq));
  const K = keep(matmul(device, h, Wk));
  const V = keep(matmul(device, h, Wv));
  const scores = keep(matmulABT(device, Q, K));
  const attn = keep(causalSoftmax(device, scores, scale));
  const ctx = keep(matmul(device, attn, V));
  const o = keep(matmul(device, ctx, Wo));
  const xa = keep(addTensors(device, x, o));
  const h2 = keep(layerNorm(device, xa, ln2g, ln2b));
  const f = keep(matmul(device, h2, Wff1));
  biasAdd(device, f, bff1);
  const fg = keep(gelu(device, f));
  const f2 = keep(matmul(device, fg, Wff2));
  biasAdd(device, f2, bff2);
  const xb = keep(addTensors(device, xa, f2));
  const xf = keep(layerNorm(device, xb, lnfg, lnfb));
  const logits = keep(matmul(device, xf, head));

  // ---- backward ----
  const probs = keep(softmax(device, logits));
  const targetBuf = createU32Buffer(device, Uint32Array.from(targets), 'targets');
  const dlogits = keep(softmaxCeBackward(device, probs, targetBuf)); // (P − onehot)/t

  const dHead = keep(matmulATB(device, xf, dlogits));
  const dxf = keep(matmulABT(device, dlogits, head));

  const lnf = layerNormBackward(device, xb, dxf, lnfg);
  keep(lnf.dx);
  keep(lnf.dgamma);
  keep(lnf.dbeta);
  const df2 = lnf.dx; // residual: xb = xa + f2

  const dWff2 = keep(matmulATB(device, fg, df2));
  const dbff2 = keep(biasBackward(device, df2));
  const dfg = keep(matmulABT(device, df2, Wff2));
  const df = keep(geluBackward(device, dfg, f));
  const dWff1 = keep(matmulATB(device, h2, df));
  const dbff1 = keep(biasBackward(device, df));
  const dh2 = keep(matmulABT(device, df, Wff1));
  const ln2 = layerNormBackward(device, xa, dh2, ln2g);
  keep(ln2.dx);
  keep(ln2.dgamma);
  keep(ln2.dbeta);
  const dxa = keep(addTensors(device, df2, ln2.dx)); // dxb + dxa_mlp

  const dWo = keep(matmulATB(device, ctx, dxa));
  const dctx = keep(matmulABT(device, dxa, Wo));

  const dattn = keep(matmulABT(device, dctx, V));
  const dV = keep(matmulATB(device, attn, dctx));
  const dscores = keep(causalSoftmaxBackward(device, attn, dattn, scale));
  const dQ = keep(matmul(device, dscores, K));
  const dK = keep(matmulATB(device, dscores, Q));

  const dWq = keep(matmulATB(device, h, dQ));
  const dWk = keep(matmulATB(device, h, dK));
  const dWv = keep(matmulATB(device, h, dV));
  const dhq = keep(matmulABT(device, dQ, Wq));
  const dhk = keep(matmulABT(device, dK, Wk));
  const dhv = keep(matmulABT(device, dV, Wv));
  const dh = keep(addTensors(device, keep(addTensors(device, dhq, dhk)), dhv));
  const ln1 = layerNormBackward(device, x, dh, ln1g);
  keep(ln1.dx);
  keep(ln1.dgamma);
  keep(ln1.dbeta);
  const dx = keep(addTensors(device, dxa, ln1.dx));

  const [
    dxArr,
    dHeadA,
    dWqA,
    dWkA,
    dWvA,
    dWoA,
    dWff1A,
    dbff1A,
    dWff2A,
    dbff2A,
    dln1gA,
    dln1bA,
    dln2gA,
    dln2bA,
    dlnfgA,
    dlnfbA,
  ] = await Promise.all([
    dx.toArray(),
    dHead.toArray(),
    dWq.toArray(),
    dWk.toArray(),
    dWv.toArray(),
    dWo.toArray(),
    dWff1.toArray(),
    dbff1.toArray(),
    dWff2.toArray(),
    dbff2.toArray(),
    ln1.dgamma.toArray(),
    ln1.dbeta.toArray(),
    ln2.dgamma.toArray(),
    ln2.dbeta.toArray(),
    lnf.dgamma.toArray(),
    lnf.dbeta.toArray(),
  ]);

  // embedding scatter-add (CPU): a token id may repeat across positions
  const dTokEmb = new Float32Array(vocab * dE);
  const dPosEmb = new Float32Array(model.cfg.blockSize * dE);
  for (let i = 0; i < t; i++) {
    const te = ids[i] * dE;
    const pe = i * dE;
    for (let c = 0; c < dE; c++) {
      const g = dxArr[i * dE + c];
      dTokEmb[te + c] += g;
      dPosEmb[pe + c] += g;
    }
  }

  for (const tn of tensors) tn.destroy();
  targetBuf.destroy();

  return {
    tokEmb: dTokEmb,
    posEmb: dPosEmb,
    'ln1.g': dln1gA,
    'ln1.b': dln1bA,
    Wq: dWqA,
    Wk: dWkA,
    Wv: dWvA,
    Wo: dWoA,
    'ln2.g': dln2gA,
    'ln2.b': dln2bA,
    Wff1: dWff1A,
    bff1: dbff1A,
    Wff2: dWff2A,
    bff2: dbff2A,
    'lnf.g': dlnfgA,
    'lnf.b': dlnfbA,
    head: dHeadA,
  };
}
