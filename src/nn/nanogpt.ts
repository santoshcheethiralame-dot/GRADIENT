// nano-GPT — a tiny char-level transformer, built in increments.
//
// This is the foundation for "gradient v2: it writes". The plan, mirroring the
// six verified phases that built the MLP engine:
//
//   ✅ Increment 1 (this file): the architecture, forward-only, on the CPU.
//        char tokenizer · token+position embeddings · one pre-LN transformer
//        block (causal single-head self-attention + GELU MLP) · final LN · LM
//        head · autoregressive generation. Verified in the self-test by the
//        defining property of a GPT — strict causality — plus softmax/LN/vocab
//        invariants. No backward yet, so nothing here needs gradient-checking.
//   ⬜ Increment 2: the backward pass (the hard part — gradients through softmax
//        attention and layer-norm), gated by numerical gradient checking exactly
//        like the MLP, then a training loop that overfits this corpus so it
//        learns to reproduce it.
//   ⬜ Increment 3: port the hot paths (QKᵀ, softmax, A·V, the projections) to
//        WGSL compute shaders — most matmuls reuse the kernels we already have;
//        the new ones are causal-masked softmax and layer-norm.
//   ⬜ Increment 4: a dashboard panel — type a prompt, watch it generate
//        character by character on the GPU, with the live attention matrix.
//
// Everything is f32/f64 CPU math here; deterministic given a seeded RNG so the
// self-test is reproducible.

import { mulberry32, gaussian } from '../data/synthetic';

// A short, original corpus — for increment 1 it only defines the character
// vocabulary and seeds generation; increment 2 will train on it.
export const NANO_CORPUS = `gradient descent walks downhill.
the network reads one character at a time, then guesses the next.
attention lets each token look back at the tokens before it, never ahead.
layer by layer, the loss melts toward zero.`;

/** Char-level tokenizer: a stable, sorted vocabulary built from a corpus. */
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
  dEmbed: number; // model width
  dFF: number; // MLP hidden width
  blockSize: number; // max context length T
}

export function defaultConfig(vocab: number): NanoGptConfig {
  return { vocab, dEmbed: 32, dFF: 128, blockSize: 32 };
}

const LN_EPS = 1e-5;

/** Row-wise layer norm: each row of x ([rows × d]) → zero mean, unit var, then
 *  affine (γ, β). Exported so the self-test can check it in isolation. */
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

/** Numerically stable softmax over a slice [off, off+n) of `x`, written in place. */
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

// A[m×k] @ B[k×n] → [m×n]
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

// tanh-approx GELU (the GPT-2 variant), applied in place.
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

/** A single-block, single-head char-level GPT. Forward + generate only. */
export class NanoGpt {
  readonly cfg: NanoGptConfig;
  // embeddings
  readonly tokEmb: Float32Array; // [vocab × dE]
  readonly posEmb: Float32Array; // [T × dE]
  // attention block (pre-LN)
  readonly ln1g: Float32Array;
  readonly ln1b: Float32Array;
  readonly Wq: Float32Array; // [dE × dE]
  readonly Wk: Float32Array;
  readonly Wv: Float32Array;
  readonly Wo: Float32Array;
  // MLP block (pre-LN)
  readonly ln2g: Float32Array;
  readonly ln2b: Float32Array;
  readonly Wff1: Float32Array; // [dE × dFF]
  readonly bff1: Float32Array;
  readonly Wff2: Float32Array; // [dFF × dE]
  readonly bff2: Float32Array;
  // head
  readonly lnfg: Float32Array;
  readonly lnfb: Float32Array;
  readonly head: Float32Array; // [dE × vocab]

  constructor(cfg: NanoGptConfig, seed = 1) {
    this.cfg = cfg;
    const { vocab, dEmbed: dE, dFF } = cfg;
    const rng = mulberry32(seed);
    const s = 0.02; // GPT-style small init
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
  }

  /** Forward over a context of t ≤ blockSize token ids. Returns logits [t × vocab]. */
  forward(ids: number[]): Float32Array {
    const { dEmbed: dE, dFF, vocab } = this.cfg;
    const t = ids.length;

    // token + position embeddings
    let x = new Float32Array(t * dE);
    for (let i = 0; i < t; i++) {
      const te = ids[i] * dE;
      const pe = i * dE;
      for (let c = 0; c < dE; c++) x[i * dE + c] = this.tokEmb[te + c] + this.posEmb[pe + c];
    }

    // ---- attention (pre-LN, single head, causal) ----
    const h = layerNormRows(x, t, dE, this.ln1g, this.ln1b);
    const Q = matmul(h, this.Wq, t, dE, dE);
    const K = matmul(h, this.Wk, t, dE, dE);
    const V = matmul(h, this.Wv, t, dE, dE);
    const scale = 1 / Math.sqrt(dE);
    const ctx = new Float32Array(t * dE);
    const scores = new Float32Array(t); // reused per query row, length ≤ t
    for (let i = 0; i < t; i++) {
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let c = 0; c < dE; c++) dot += Q[i * dE + c] * K[j * dE + c];
        scores[j] = dot * scale;
      }
      softmaxRow(scores, 0, i + 1); // causal: only keys 0..i
      for (let j = 0; j <= i; j++) {
        const a = scores[j];
        for (let c = 0; c < dE; c++) ctx[i * dE + c] += a * V[j * dE + c];
      }
    }
    const o = matmul(ctx, this.Wo, t, dE, dE);
    for (let i = 0; i < t * dE; i++) x[i] += o[i]; // residual

    // ---- MLP (pre-LN) ----
    const h2 = layerNormRows(x, t, dE, this.ln2g, this.ln2b);
    const f = matmul(h2, this.Wff1, t, dE, dFF);
    for (let i = 0; i < t; i++) for (let c = 0; c < dFF; c++) f[i * dFF + c] += this.bff1[c];
    geluInPlace(f);
    const f2 = matmul(f, this.Wff2, t, dFF, dE);
    for (let i = 0; i < t; i++) for (let c = 0; c < dE; c++) f2[i * dE + c] += this.bff2[c];
    for (let i = 0; i < t * dE; i++) x[i] += f2[i]; // residual

    // ---- head ----
    const xf = layerNormRows(x, t, dE, this.lnfg, this.lnfb);
    const logits = matmul(xf, this.head, t, dE, vocab);
    return logits;
  }

  /** Autoregressively sample `count` tokens after the prompt ids. temperature 0
   *  = greedy argmax. Returns the full id sequence (prompt + generated). */
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
        // inverse-CDF sample
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
}
