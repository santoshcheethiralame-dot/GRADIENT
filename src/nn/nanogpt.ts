import { mulberry32, gaussian } from '../data/synthetic';

export const NANO_CORPUS = `gradient descent walks downhill.
the network reads one character at a time, then guesses the next.
attention lets each token look back at the tokens before it, never ahead.
layer by layer, the loss melts toward zero.`;

export class CharTokenizer {
  readonly chars: string[];
  readonly vocab: number;
  private readonly stoi: Map<string, number>;

  constructor(corpus: string) {
    this.chars = Array.from(new Set(corpus.split(''))).sort();
    this.vocab = this.chars.length;
    this.stoi = new Map(this.chars.map((c, i) => [c, i]));
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const ch of text) {
      const id = this.stoi.get(ch);
      if (id !== undefined) out.push(id);
    }
    return out;
  }

  decode(ids: number[]): string {
    let s = '';
    for (const id of ids) s += this.chars[id] ?? '';
    return s;
  }
}

export interface NanoGptConfig {
  vocab: number;
  dEmbed: number;
  dFF: number;
  blockSize: number;
}

export function defaultConfig(vocab: number): NanoGptConfig {
  return { vocab, dEmbed: 32, dFF: 128, blockSize: 32 };
}

const LN_EPS = 1e-5;

export function layerNormRows(
  x: Float32Array,
  rows: number,
  d: number,
  gamma: Float32Array,
  beta: Float32Array,
): Float32Array {
  const out = new Float32Array(rows * d);
  for (let r = 0; r < rows; r++) {
    const off = r * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[off + i];
    mean /= d;
    let varc = 0;
    for (let i = 0; i < d; i++) {
      const v = x[off + i] - mean;
      varc += v * v;
    }
    varc /= d;
    const inv = 1 / Math.sqrt(varc + LN_EPS);
    for (let i = 0; i < d; i++) out[off + i] = (x[off + i] - mean) * inv * gamma[i] + beta[i];
  }
  return out;
}

export function softmaxRow(x: Float32Array, off: number, n: number): void {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[off + i] > max) max = x[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[off + i] - max);
    x[off + i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < n; i++) x[off + i] *= inv;
}

function matmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = a[i * k + p];
      if (av === 0) continue;
      const brow = p * n;
      const orow = i * n;
      for (let j = 0; j < n; j++) out[orow + j] += av * b[brow + j];
    }
  }
  return out;
}

function geluInPlace(x: Float32Array): void {
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
}

function randn(n: number, std: number, rng: () => number): Float32Array {
  const o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = gaussian(rng) * std;
  return o;
}

function matmulATB(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
  const out = new Float32Array(K * N);
  for (let i = 0; i < M; i++) {
    for (let p = 0; p < K; p++) {
      const av = a[i * K + p];
      if (av === 0) continue;
      const brow = i * N;
      const orow = p * N;
      for (let q = 0; q < N; q++) out[orow + q] += av * b[brow + q];
    }
  }
  return out;
}

function matmulABT(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
  const out = new Float32Array(M * K);
  for (let i = 0; i < M; i++) {
    const arow = i * N;
    for (let p = 0; p < K; p++) {
      const brow = p * N;
      let s = 0;
      for (let j = 0; j < N; j++) s += a[arow + j] * b[brow + j];
      out[i * K + p] = s;
    }
  }
  return out;
}

function colSum(m: Float32Array, rows: number, d: number): Float32Array {
  const o = new Float32Array(d);
  for (let r = 0; r < rows; r++) for (let i = 0; i < d; i++) o[i] += m[r * d + i];
  return o;
}

function addInto(dst: Float32Array, src: Float32Array): void {
  for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

function geluGrad(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const u = c * (x + 0.044715 * x * x * x);
  const t = Math.tanh(u);
  const du = c * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

function layerNormBackward(
  dy: Float32Array,
  xIn: Float32Array,
  gamma: Float32Array,
  rows: number,
  d: number,
): { dx: Float32Array; dgamma: Float32Array; dbeta: Float32Array } {
  const dx = new Float32Array(rows * d);
  const dgamma = new Float32Array(d);
  const dbeta = new Float32Array(d);
  const xhat = new Float32Array(d);
  const dxhat = new Float32Array(d);
  for (let r = 0; r < rows; r++) {
    const off = r * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += xIn[off + i];
    mean /= d;
    let varc = 0;
    for (let i = 0; i < d; i++) {
      const v = xIn[off + i] - mean;
      varc += v * v;
    }
    varc /= d;
    const inv = 1 / Math.sqrt(varc + LN_EPS);
    let sumDxhat = 0;
    let sumDxhatXhat = 0;
    for (let i = 0; i < d; i++) {
      const xh = (xIn[off + i] - mean) * inv;
      xhat[i] = xh;
      const dyi = dy[off + i];
      dbeta[i] += dyi;
      dgamma[i] += dyi * xh;
      const dxh = dyi * gamma[i];
      dxhat[i] = dxh;
      sumDxhat += dxh;
      sumDxhatXhat += dxh * xh;
    }
    for (let i = 0; i < d; i++)
      dx[off + i] = inv * (dxhat[i] - sumDxhat / d - xhat[i] * (sumDxhatXhat / d));
  }
  return { dx, dgamma, dbeta };
}

export interface NanoGptCache {
  ids: number[];
  t: number;
  x: Float32Array;
  h: Float32Array;
  Q: Float32Array;
  K: Float32Array;
  V: Float32Array;
  attn: Float32Array;
  ctx: Float32Array;
  xa: Float32Array;
  h2: Float32Array;
  f: Float32Array;
  fg: Float32Array;
  xb: Float32Array;
  xf: Float32Array;
  logits: Float32Array;
}

export class NanoGpt {
  readonly cfg: NanoGptConfig;
  readonly tokEmb: Float32Array;
  readonly posEmb: Float32Array;
  readonly ln1g: Float32Array;
  readonly ln1b: Float32Array;
  readonly Wq: Float32Array;
  readonly Wk: Float32Array;
  readonly Wv: Float32Array;
  readonly Wo: Float32Array;
  readonly ln2g: Float32Array;
  readonly ln2b: Float32Array;
  readonly Wff1: Float32Array;
  readonly bff1: Float32Array;
  readonly Wff2: Float32Array;
  readonly bff2: Float32Array;
  readonly lnfg: Float32Array;
  readonly lnfb: Float32Array;
  readonly head: Float32Array;

  readonly dTokEmb: Float32Array;
  readonly dPosEmb: Float32Array;
  readonly dLn1g: Float32Array;
  readonly dLn1b: Float32Array;
  readonly dWq: Float32Array;
  readonly dWk: Float32Array;
  readonly dWv: Float32Array;
  readonly dWo: Float32Array;
  readonly dLn2g: Float32Array;
  readonly dLn2b: Float32Array;
  readonly dWff1: Float32Array;
  readonly dbff1: Float32Array;
  readonly dWff2: Float32Array;
  readonly dbff2: Float32Array;
  readonly dLnfg: Float32Array;
  readonly dLnfb: Float32Array;
  readonly dHead: Float32Array;

  constructor(cfg: NanoGptConfig, seed = 1, initStd = 0.02) {
    this.cfg = cfg;
    const { vocab, dEmbed: dE, dFF } = cfg;
    const rng = mulberry32(seed);
    const s = initStd;
    this.tokEmb = randn(vocab * dE, s, rng);
    this.posEmb = randn(cfg.blockSize * dE, s, rng);
    this.ln1g = new Float32Array(dE).fill(1);
    this.ln1b = new Float32Array(dE);
    this.Wq = randn(dE * dE, s, rng);
    this.Wk = randn(dE * dE, s, rng);
    this.Wv = randn(dE * dE, s, rng);
    this.Wo = randn(dE * dE, s, rng);
    this.ln2g = new Float32Array(dE).fill(1);
    this.ln2b = new Float32Array(dE);
    this.Wff1 = randn(dE * dFF, s, rng);
    this.bff1 = new Float32Array(dFF);
    this.Wff2 = randn(dFF * dE, s, rng);
    this.bff2 = new Float32Array(dE);
    this.lnfg = new Float32Array(dE).fill(1);
    this.lnfb = new Float32Array(dE);
    this.head = randn(dE * vocab, s, rng);

    this.dTokEmb = new Float32Array(vocab * dE);
    this.dPosEmb = new Float32Array(cfg.blockSize * dE);
    this.dLn1g = new Float32Array(dE);
    this.dLn1b = new Float32Array(dE);
    this.dWq = new Float32Array(dE * dE);
    this.dWk = new Float32Array(dE * dE);
    this.dWv = new Float32Array(dE * dE);
    this.dWo = new Float32Array(dE * dE);
    this.dLn2g = new Float32Array(dE);
    this.dLn2b = new Float32Array(dE);
    this.dWff1 = new Float32Array(dE * dFF);
    this.dbff1 = new Float32Array(dFF);
    this.dWff2 = new Float32Array(dFF * dE);
    this.dbff2 = new Float32Array(dE);
    this.dLnfg = new Float32Array(dE);
    this.dLnfb = new Float32Array(dE);
    this.dHead = new Float32Array(dE * vocab);
  }

  private run(ids: number[], cache?: NanoGptCache): Float32Array {
    const { dEmbed: dE, dFF, vocab } = this.cfg;
    const t = ids.length;

    const x = new Float32Array(t * dE);
    for (let i = 0; i < t; i++) {
      const te = ids[i] * dE;
      const pe = i * dE;
      for (let c = 0; c < dE; c++) x[i * dE + c] = this.tokEmb[te + c] + this.posEmb[pe + c];
    }

    const h = layerNormRows(x, t, dE, this.ln1g, this.ln1b);
    const Q = matmul(h, this.Wq, t, dE, dE);
    const K = matmul(h, this.Wk, t, dE, dE);
    const V = matmul(h, this.Wv, t, dE, dE);
    const scale = 1 / Math.sqrt(dE);
    const ctx = new Float32Array(t * dE);
    const attn = cache ? new Float32Array(t * t) : null;
    const scores = new Float32Array(t);
    for (let i = 0; i < t; i++) {
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let c = 0; c < dE; c++) dot += Q[i * dE + c] * K[j * dE + c];
        scores[j] = dot * scale;
      }
      softmaxRow(scores, 0, i + 1);
      for (let j = 0; j <= i; j++) {
        const a = scores[j];
        if (attn) attn[i * t + j] = a;
        for (let c = 0; c < dE; c++) ctx[i * dE + c] += a * V[j * dE + c];
      }
    }
    const o = matmul(ctx, this.Wo, t, dE, dE);
    const xa = new Float32Array(t * dE);
    for (let i = 0; i < t * dE; i++) xa[i] = x[i] + o[i];

    const h2 = layerNormRows(xa, t, dE, this.ln2g, this.ln2b);
    const f = matmul(h2, this.Wff1, t, dE, dFF);
    for (let i = 0; i < t; i++) for (let c = 0; c < dFF; c++) f[i * dFF + c] += this.bff1[c];
    const fg = f.slice();
    geluInPlace(fg);
    const f2 = matmul(fg, this.Wff2, t, dFF, dE);
    for (let i = 0; i < t; i++) for (let c = 0; c < dE; c++) f2[i * dE + c] += this.bff2[c];
    const xb = new Float32Array(t * dE);
    for (let i = 0; i < t * dE; i++) xb[i] = xa[i] + f2[i];

    const xf = layerNormRows(xb, t, dE, this.lnfg, this.lnfb);
    const logits = matmul(xf, this.head, t, dE, vocab);

    if (cache) {
      cache.ids = ids;
      cache.t = t;
      cache.x = x;
      cache.h = h;
      cache.Q = Q;
      cache.K = K;
      cache.V = V;
      cache.attn = attn as Float32Array;
      cache.ctx = ctx;
      cache.xa = xa;
      cache.h2 = h2;
      cache.f = f;
      cache.fg = fg;
      cache.xb = xb;
      cache.xf = xf;
      cache.logits = logits;
    }
    return logits;
  }

  forward(ids: number[]): Float32Array {
    return this.run(ids);
  }

  forwardCache(ids: number[]): NanoGptCache {
    const cache = {} as NanoGptCache;
    this.run(ids, cache);
    return cache;
  }

  generate(promptIds: number[], count: number, temperature: number, rng: () => number): number[] {
    const { blockSize, vocab } = this.cfg;
    const ids = promptIds.slice();
    for (let n = 0; n < count; n++) {
      const ctx = ids.slice(Math.max(0, ids.length - blockSize));
      const logits = this.forward(ctx);
      const last = (ctx.length - 1) * vocab;
      let next: number;
      if (temperature <= 0) {
        let best = 0;
        for (let v = 1; v < vocab; v++) if (logits[last + v] > logits[last + best]) best = v;
        next = best;
      } else {
        const probs = new Float32Array(vocab);
        for (let v = 0; v < vocab; v++) probs[v] = logits[last + v] / temperature;
        softmaxRow(probs, 0, vocab);
        let r = rng();
        next = vocab - 1;
        for (let v = 0; v < vocab; v++) {
          r -= probs[v];
          if (r <= 0) {
            next = v;
            break;
          }
        }
      }
      ids.push(next);
    }
    return ids;
  }

  static crossEntropy(
    logits: Float32Array,
    targets: number[],
    t: number,
    vocab: number,
  ): { loss: number; probs: Float32Array } {
    const probs = new Float32Array(t * vocab);
    let loss = 0;
    for (let i = 0; i < t; i++) {
      const off = i * vocab;
      for (let v = 0; v < vocab; v++) probs[off + v] = logits[off + v];
      softmaxRow(probs, off, vocab);
      loss += -Math.log(Math.max(probs[off + targets[i]], 1e-12));
    }
    return { loss: loss / t, probs };
  }

  forwardLoss(ids: number[], targets: number[]): number {
    const logits = this.run(ids);
    return NanoGpt.crossEntropy(logits, targets, ids.length, this.cfg.vocab).loss;
  }

  params(): Array<{ name: string; w: Float32Array; g: Float32Array }> {
    return [
      { name: 'tokEmb', w: this.tokEmb, g: this.dTokEmb },
      { name: 'posEmb', w: this.posEmb, g: this.dPosEmb },
      { name: 'ln1.g', w: this.ln1g, g: this.dLn1g },
      { name: 'ln1.b', w: this.ln1b, g: this.dLn1b },
      { name: 'Wq', w: this.Wq, g: this.dWq },
      { name: 'Wk', w: this.Wk, g: this.dWk },
      { name: 'Wv', w: this.Wv, g: this.dWv },
      { name: 'Wo', w: this.Wo, g: this.dWo },
      { name: 'ln2.g', w: this.ln2g, g: this.dLn2g },
      { name: 'ln2.b', w: this.ln2b, g: this.dLn2b },
      { name: 'Wff1', w: this.Wff1, g: this.dWff1 },
      { name: 'bff1', w: this.bff1, g: this.dbff1 },
      { name: 'Wff2', w: this.Wff2, g: this.dWff2 },
      { name: 'bff2', w: this.bff2, g: this.dbff2 },
      { name: 'lnf.g', w: this.lnfg, g: this.dLnfg },
      { name: 'lnf.b', w: this.lnfb, g: this.dLnfb },
      { name: 'head', w: this.head, g: this.dHead },
    ];
  }

  zeroGrad(): void {
    for (const p of this.params()) p.g.fill(0);
  }

  backward(c: NanoGptCache, targets: number[]): void {
    const { dEmbed: dE, dFF, vocab } = this.cfg;
    const t = c.t;

    const { probs } = NanoGpt.crossEntropy(c.logits, targets, t, vocab);
    const dlogits = new Float32Array(t * vocab);
    for (let i = 0; i < t; i++) {
      const off = i * vocab;
      for (let v = 0; v < vocab; v++) dlogits[off + v] = probs[off + v] / t;
      dlogits[off + targets[i]] -= 1 / t;
    }

    addInto(this.dHead, matmulATB(c.xf, dlogits, t, dE, vocab));
    const dxf = matmulABT(dlogits, this.head, t, dE, vocab);

    const lnf = layerNormBackward(dxf, c.xb, this.lnfg, t, dE);
    addInto(this.dLnfg, lnf.dgamma);
    addInto(this.dLnfb, lnf.dbeta);

    const dxa = lnf.dx.slice();
    const df2 = lnf.dx;

    addInto(this.dWff2, matmulATB(c.fg, df2, t, dFF, dE));
    addInto(this.dbff2, colSum(df2, t, dE));
    const dfg = matmulABT(df2, this.Wff2, t, dFF, dE);
    const df = new Float32Array(t * dFF);
    for (let i = 0; i < t * dFF; i++) df[i] = dfg[i] * geluGrad(c.f[i]);
    addInto(this.dWff1, matmulATB(c.h2, df, t, dE, dFF));
    addInto(this.dbff1, colSum(df, t, dFF));
    const dh2 = matmulABT(df, this.Wff1, t, dE, dFF);
    const ln2 = layerNormBackward(dh2, c.xa, this.ln2g, t, dE);
    addInto(this.dLn2g, ln2.dgamma);
    addInto(this.dLn2b, ln2.dbeta);
    for (let i = 0; i < t * dE; i++) dxa[i] += ln2.dx[i];

    const dx = dxa.slice();
    const dout = dxa;

    addInto(this.dWo, matmulATB(c.ctx, dout, t, dE, dE));
    const dctx = matmulABT(dout, this.Wo, t, dE, dE);

    const scale = 1 / Math.sqrt(dE);
    const dQ = new Float32Array(t * dE);
    const dK = new Float32Array(t * dE);
    const dV = new Float32Array(t * dE);
    for (let i = 0; i < t; i++) {
      const datt = new Float32Array(i + 1);
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let cc = 0; cc < dE; cc++) dot += dctx[i * dE + cc] * c.V[j * dE + cc];
        datt[j] = dot;
        const a = c.attn[i * t + j];
        for (let cc = 0; cc < dE; cc++) dV[j * dE + cc] += a * dctx[i * dE + cc];
      }
      let sumd = 0;
      for (let j = 0; j <= i; j++) sumd += datt[j] * c.attn[i * t + j];
      for (let j = 0; j <= i; j++) {
        const ds = c.attn[i * t + j] * (datt[j] - sumd) * scale;
        for (let cc = 0; cc < dE; cc++) {
          dQ[i * dE + cc] += ds * c.K[j * dE + cc];
          dK[j * dE + cc] += ds * c.Q[i * dE + cc];
        }
      }
    }

    addInto(this.dWq, matmulATB(c.h, dQ, t, dE, dE));
    addInto(this.dWk, matmulATB(c.h, dK, t, dE, dE));
    addInto(this.dWv, matmulATB(c.h, dV, t, dE, dE));
    const dh = matmulABT(dQ, this.Wq, t, dE, dE);
    const dhk = matmulABT(dK, this.Wk, t, dE, dE);
    const dhv = matmulABT(dV, this.Wv, t, dE, dE);
    for (let i = 0; i < t * dE; i++) dh[i] += dhk[i] + dhv[i];

    const ln1 = layerNormBackward(dh, c.x, this.ln1g, t, dE);
    addInto(this.dLn1g, ln1.dgamma);
    addInto(this.dLn1b, ln1.dbeta);
    for (let i = 0; i < t * dE; i++) dx[i] += ln1.dx[i];

    for (let i = 0; i < t; i++) {
      const ti = c.ids[i] * dE;
      const pi = i * dE;
      for (let cc = 0; cc < dE; cc++) {
        const g = dx[i * dE + cc];
        this.dTokEmb[ti + cc] += g;
        this.dPosEmb[pi + cc] += g;
      }
    }
  }
}
