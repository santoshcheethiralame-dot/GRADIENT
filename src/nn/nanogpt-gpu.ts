import { GpuTensor } from '../gpu/tensor';
import {
  matmul,
  matmulABT,
  biasAdd,
  layerNorm,
  causalSoftmax,
  gelu,
  addTensors,
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
