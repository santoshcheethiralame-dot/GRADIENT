import { GpuTensor } from '../gpu/tensor';
import {
  matmul,
  matmulATB,
  matmulABT,
  biasAdd,
  biasBackward,
  softmax,
  softmaxCeBackward,
  crossEntropy,
  adamStep,
  layerNorm,
  layerNormBackward,
  causalSoftmax,
  causalSoftmaxBackward,
  gelu,
  geluBackward,
  addTensors,
  createU32Buffer,
  sliceCols,
  pasteCols,
} from '../gpu/ops';
import { NanoGpt } from './nanogpt';

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
  const H = model.cfg.nHeads ?? 1;
  const dH = dE / H;
  const ascale = 1 / Math.sqrt(dH);
  const ctx = keep(GpuTensor.zeros(device, [t, dE]));
  for (let hd = 0; hd < H; hd++) {
    const c0 = hd * dH;
    const Qh = keep(sliceCols(device, Q, c0, dH));
    const Kh = keep(sliceCols(device, K, c0, dH));
    const Vh = keep(sliceCols(device, V, c0, dH));
    const sh = keep(matmulABT(device, Qh, Kh));
    const ah = keep(causalSoftmax(device, sh, ascale));
    const ch = keep(matmul(device, ah, Vh));
    pasteCols(device, ctx, ch, c0);
  }
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

export async function nanoGptGpuBackward(
  device: GPUDevice,
  model: NanoGpt,
  ids: number[],
  targets: number[],
): Promise<Record<string, Float32Array>> {
  const { dEmbed: dE, dFF, vocab } = model.cfg;
  const t = ids.length;
  const H = model.cfg.nHeads ?? 1;
  const dH = dE / H;
  const ascale = 1 / Math.sqrt(dH);

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

  const h = keep(layerNorm(device, x, ln1g, ln1b));
  const Q = keep(matmul(device, h, Wq));
  const K = keep(matmul(device, h, Wk));
  const V = keep(matmul(device, h, Wv));
  const ahs: GpuTensor[] = [];
  const ctx = keep(GpuTensor.zeros(device, [t, dE]));
  for (let hd = 0; hd < H; hd++) {
    const c0 = hd * dH;
    const Qh = keep(sliceCols(device, Q, c0, dH));
    const Kh = keep(sliceCols(device, K, c0, dH));
    const Vh = keep(sliceCols(device, V, c0, dH));
    const sh = keep(matmulABT(device, Qh, Kh));
    const ah = keep(causalSoftmax(device, sh, ascale));
    ahs.push(ah);
    const ch = keep(matmul(device, ah, Vh));
    pasteCols(device, ctx, ch, c0);
  }
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

  const probs = keep(softmax(device, logits));
  const targetBuf = createU32Buffer(device, Uint32Array.from(targets), 'targets');
  const dlogits = keep(softmaxCeBackward(device, probs, targetBuf));

  const dHead = keep(matmulATB(device, xf, dlogits));
  const dxf = keep(matmulABT(device, dlogits, head));

  const lnf = layerNormBackward(device, xb, dxf, lnfg);
  keep(lnf.dx);
  keep(lnf.dgamma);
  keep(lnf.dbeta);
  const df2 = lnf.dx;

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
  const dxa = keep(addTensors(device, df2, ln2.dx));

  const dWo = keep(matmulATB(device, ctx, dxa));
  const dctx = keep(matmulABT(device, dxa, Wo));

  const dQ = keep(GpuTensor.zeros(device, [t, dE]));
  const dK = keep(GpuTensor.zeros(device, [t, dE]));
  const dV = keep(GpuTensor.zeros(device, [t, dE]));
  for (let hd = 0; hd < H; hd++) {
    const c0 = hd * dH;
    const Qh = keep(sliceCols(device, Q, c0, dH));
    const Kh = keep(sliceCols(device, K, c0, dH));
    const Vh = keep(sliceCols(device, V, c0, dH));
    const dch = keep(sliceCols(device, dctx, c0, dH));
    const ah = ahs[hd];
    const dattnH = keep(matmulABT(device, dch, Vh));
    const dVh = keep(matmulATB(device, ah, dch));
    const dscoresH = keep(causalSoftmaxBackward(device, ah, dattnH, ascale));
    const dQh = keep(matmul(device, dscoresH, Kh));
    const dKh = keep(matmulATB(device, dscoresH, Qh));
    pasteCols(device, dQ, dQh, c0);
    pasteCols(device, dK, dKh, c0);
    pasteCols(device, dV, dVh, c0);
  }

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

const BLOCK_PARAMS = [
  'ln1g',
  'ln1b',
  'Wq',
  'Wk',
  'Wv',
  'Wo',
  'ln2g',
  'ln2b',
  'Wff1',
  'bff1',
  'Wff2',
  'bff2',
  'lnfg',
  'lnfb',
  'head',
] as const;

const B1 = 0.9;
const B2 = 0.999;
const EPS = 1e-8;

export class GpuNanoGpt {
  readonly device: GPUDevice;
  readonly cfg: NanoGpt['cfg'];
  private readonly p: Record<string, GpuTensor> = {};
  private readonly m: Record<string, GpuTensor> = {};
  private readonly v: Record<string, GpuTensor> = {};
  private readonly tokEmb: Float32Array;
  private readonly posEmb: Float32Array;
  private readonly mTok: Float32Array;
  private readonly vTok: Float32Array;
  private readonly mPos: Float32Array;
  private readonly vPos: Float32Array;
  private adamT = 0;

  constructor(device: GPUDevice, model: NanoGpt) {
    this.device = device;
    this.cfg = model.cfg;
    const { dEmbed: dE, dFF, vocab, blockSize } = model.cfg;
    const shape: Record<string, number[]> = {
      ln1g: [dE],
      ln1b: [dE],
      Wq: [dE, dE],
      Wk: [dE, dE],
      Wv: [dE, dE],
      Wo: [dE, dE],
      ln2g: [dE],
      ln2b: [dE],
      Wff1: [dE, dFF],
      bff1: [dFF],
      Wff2: [dFF, dE],
      bff2: [dE],
      lnfg: [dE],
      lnfb: [dE],
      head: [dE, vocab],
    };
    for (const name of BLOCK_PARAMS) {
      const src = (model as unknown as Record<string, Float32Array>)[name];
      this.p[name] = GpuTensor.fromArray(device, src.slice(), shape[name]);
      this.m[name] = GpuTensor.zeros(device, shape[name]);
      this.v[name] = GpuTensor.zeros(device, shape[name]);
    }
    this.tokEmb = model.tokEmb.slice();
    this.posEmb = model.posEmb.slice();
    this.mTok = new Float32Array(vocab * dE);
    this.vTok = new Float32Array(vocab * dE);
    this.mPos = new Float32Array(blockSize * dE);
    this.vPos = new Float32Array(blockSize * dE);
  }

  async step(ids: number[], targets: number[], lr: number): Promise<number> {
    const device = this.device;
    const { dEmbed: dE } = this.cfg;
    const t = ids.length;
    const H = this.cfg.nHeads ?? 1;
    const dH = dE / H;
    const ascale = 1 / Math.sqrt(dH);
    const P = this.p;
    const scratch: GpuTensor[] = [];
    const k = <T extends GpuTensor>(g: T): T => {
      scratch.push(g);
      return g;
    };

    const xData = new Float32Array(t * dE);
    for (let i = 0; i < t; i++) {
      const te = ids[i] * dE;
      const pe = i * dE;
      for (let c = 0; c < dE; c++) xData[i * dE + c] = this.tokEmb[te + c] + this.posEmb[pe + c];
    }
    const x = k(GpuTensor.fromArray(device, xData, [t, dE]));

    const h = k(layerNorm(device, x, P.ln1g, P.ln1b));
    const Q = k(matmul(device, h, P.Wq));
    const K = k(matmul(device, h, P.Wk));
    const V = k(matmul(device, h, P.Wv));
    const ahs: GpuTensor[] = [];
    const ctx = k(GpuTensor.zeros(device, [t, dE]));
    for (let hd = 0; hd < H; hd++) {
      const c0 = hd * dH;
      const Qh = k(sliceCols(device, Q, c0, dH));
      const Kh = k(sliceCols(device, K, c0, dH));
      const Vh = k(sliceCols(device, V, c0, dH));
      const sh = k(matmulABT(device, Qh, Kh));
      const ah = k(causalSoftmax(device, sh, ascale));
      ahs.push(ah);
      const ch = k(matmul(device, ah, Vh));
      pasteCols(device, ctx, ch, c0);
    }
    const o = k(matmul(device, ctx, P.Wo));
    const xa = k(addTensors(device, x, o));
    const h2 = k(layerNorm(device, xa, P.ln2g, P.ln2b));
    const f = k(matmul(device, h2, P.Wff1));
    biasAdd(device, f, P.bff1);
    const fg = k(gelu(device, f));
    const f2 = k(matmul(device, fg, P.Wff2));
    biasAdd(device, f2, P.bff2);
    const xb = k(addTensors(device, xa, f2));
    const xf = k(layerNorm(device, xb, P.lnfg, P.lnfb));
    const logits = k(matmul(device, xf, P.head));

    const probs = k(softmax(device, logits));
    const tb = createU32Buffer(device, Uint32Array.from(targets), 'targets');
    const dlogits = k(softmaxCeBackward(device, probs, tb));
    const dHead = k(matmulATB(device, xf, dlogits));
    const dxf = k(matmulABT(device, dlogits, P.head));
    const lnf = layerNormBackward(device, xb, dxf, P.lnfg);
    k(lnf.dx);
    k(lnf.dgamma);
    k(lnf.dbeta);
    const df2 = lnf.dx;
    const dWff2 = k(matmulATB(device, fg, df2));
    const dbff2 = k(biasBackward(device, df2));
    const dfg = k(matmulABT(device, df2, P.Wff2));
    const df = k(geluBackward(device, dfg, f));
    const dWff1 = k(matmulATB(device, h2, df));
    const dbff1 = k(biasBackward(device, df));
    const dh2 = k(matmulABT(device, df, P.Wff1));
    const ln2 = layerNormBackward(device, xa, dh2, P.ln2g);
    k(ln2.dx);
    k(ln2.dgamma);
    k(ln2.dbeta);
    const dxa = k(addTensors(device, df2, ln2.dx));
    const dWo = k(matmulATB(device, ctx, dxa));
    const dctx = k(matmulABT(device, dxa, P.Wo));
    const dQ = k(GpuTensor.zeros(device, [t, dE]));
    const dK = k(GpuTensor.zeros(device, [t, dE]));
    const dV = k(GpuTensor.zeros(device, [t, dE]));
    for (let hd = 0; hd < H; hd++) {
      const c0 = hd * dH;
      const Qh = k(sliceCols(device, Q, c0, dH));
      const Kh = k(sliceCols(device, K, c0, dH));
      const Vh = k(sliceCols(device, V, c0, dH));
      const dch = k(sliceCols(device, dctx, c0, dH));
      const ah = ahs[hd];
      const dattnH = k(matmulABT(device, dch, Vh));
      const dVh = k(matmulATB(device, ah, dch));
      const dscoresH = k(causalSoftmaxBackward(device, ah, dattnH, ascale));
      const dQh = k(matmul(device, dscoresH, Kh));
      const dKh = k(matmulATB(device, dscoresH, Qh));
      pasteCols(device, dQ, dQh, c0);
      pasteCols(device, dK, dKh, c0);
      pasteCols(device, dV, dVh, c0);
    }
    const dWq = k(matmulATB(device, h, dQ));
    const dWk = k(matmulATB(device, h, dK));
    const dWv = k(matmulATB(device, h, dV));
    const dhq = k(matmulABT(device, dQ, P.Wq));
    const dhk = k(matmulABT(device, dK, P.Wk));
    const dhv = k(matmulABT(device, dV, P.Wv));
    const dh = k(addTensors(device, k(addTensors(device, dhq, dhk)), dhv));
    const ln1 = layerNormBackward(device, x, dh, P.ln1g);
    k(ln1.dx);
    k(ln1.dgamma);
    k(ln1.dbeta);
    const dx = k(addTensors(device, dxa, ln1.dx));

    this.adamT++;
    const bc1 = 1 - Math.pow(B1, this.adamT);
    const bc2 = 1 - Math.pow(B2, this.adamT);
    const cfg = { lr, beta1: B1, beta2: B2, eps: EPS, bc1, bc2 };
    const grad: Record<string, GpuTensor> = {
      head: dHead,
      Wq: dWq,
      Wk: dWk,
      Wv: dWv,
      Wo: dWo,
      Wff1: dWff1,
      bff1: dbff1,
      Wff2: dWff2,
      bff2: dbff2,
      ln1g: ln1.dgamma,
      ln1b: ln1.dbeta,
      ln2g: ln2.dgamma,
      ln2b: ln2.dbeta,
      lnfg: lnf.dgamma,
      lnfb: lnf.dbeta,
    };
    for (const name of BLOCK_PARAMS) {
      adamStep(device, this.p[name], grad[name], this.m[name], this.v[name], cfg);
    }

    const lossesT = k(crossEntropy(device, probs, tb));
    const [lossArr, dxArr] = await Promise.all([lossesT.toArray(), dx.toArray()]);
    let loss = 0;
    for (let i = 0; i < t; i++) loss += lossArr[i];
    loss /= t;

    const { vocab, blockSize } = this.cfg;
    const dTok = new Float32Array(vocab * dE);
    const dPos = new Float32Array(blockSize * dE);
    for (let i = 0; i < t; i++) {
      const te = ids[i] * dE;
      const pe = i * dE;
      for (let c = 0; c < dE; c++) {
        const gg = dxArr[i * dE + c];
        dTok[te + c] += gg;
        dPos[pe + c] += gg;
      }
    }
    adamCpu(this.tokEmb, dTok, this.mTok, this.vTok, lr, bc1, bc2);
    adamCpu(this.posEmb, dPos, this.mPos, this.vPos, lr, bc1, bc2);

    for (const g of scratch) g.destroy();
    tb.destroy();
    return loss;
  }

  async syncTo(model: NanoGpt): Promise<void> {
    model.tokEmb.set(this.tokEmb);
    model.posEmb.set(this.posEmb);
    const arrs = await Promise.all(BLOCK_PARAMS.map((name) => this.p[name].toArray()));
    BLOCK_PARAMS.forEach((name, i) => {
      (model as unknown as Record<string, Float32Array>)[name].set(arrs[i]);
    });
  }

  async attention(ids: number[]): Promise<Float32Array> {
    const device = this.device;
    const { dEmbed: dE } = this.cfg;
    const t = ids.length;
    const H = this.cfg.nHeads ?? 1;
    const dH = dE / H;
    const ascale = 1 / Math.sqrt(dH);
    const P = this.p;
    const scratch: GpuTensor[] = [];
    const k = <T extends GpuTensor>(g: T): T => {
      scratch.push(g);
      return g;
    };
    const xData = new Float32Array(t * dE);
    for (let i = 0; i < t; i++) {
      const te = ids[i] * dE;
      const pe = i * dE;
      for (let c = 0; c < dE; c++) xData[i * dE + c] = this.tokEmb[te + c] + this.posEmb[pe + c];
    }
    const x = k(GpuTensor.fromArray(device, xData, [t, dE]));
    const h = k(layerNorm(device, x, P.ln1g, P.ln1b));
    const Q = k(matmul(device, h, P.Wq));
    const K = k(matmul(device, h, P.Wk));
    const Qh = k(sliceCols(device, Q, 0, dH));
    const Kh = k(sliceCols(device, K, 0, dH));
    const scores = k(matmulABT(device, Qh, Kh));
    const attn = k(causalSoftmax(device, scores, ascale));
    const arr = await attn.toArray();
    for (const g of scratch) g.destroy();
    return arr;
  }

  destroy(): void {
    for (const name of BLOCK_PARAMS) {
      this.p[name].destroy();
      this.m[name].destroy();
      this.v[name].destroy();
    }
  }
}

function adamCpu(
  w: Float32Array,
  g: Float32Array,
  m: Float32Array,
  v: Float32Array,
  lr: number,
  bc1: number,
  bc2: number,
): void {
  for (let i = 0; i < w.length; i++) {
    m[i] = B1 * m[i] + (1 - B1) * g[i];
    v[i] = B2 * v[i] + (1 - B2) * g[i] * g[i];
    w[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + EPS);
  }
}
