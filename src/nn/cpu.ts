// CPU training engine — a 2-layer MLP whose forward/backward reuse the f64
// reference ops (the same code that serves as the GPU oracle, so it's already
// gradient-checked). Used as a fallback when WebGPU is unavailable, so the demo
// still trains everywhere — just slower.

import * as ref from './reference';

function adamUpdate(
  w: Float32Array,
  g: Float32Array,
  m: Float32Array,
  v: Float32Array,
  lr: number,
  b1: number,
  b2: number,
  eps: number,
  bc1: number,
  bc2: number,
): void {
  for (let i = 0; i < w.length; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    w[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
  }
}

export interface CpuMlpInit {
  W1: Float32Array;
  b1: Float32Array;
  W2: Float32Array;
  b2: Float32Array;
}

export class CpuMlp {
  readonly W1: Float32Array;
  readonly b1: Float32Array;
  readonly W2: Float32Array;
  readonly b2: Float32Array;

  private readonly mW1: Float32Array;
  private readonly vW1: Float32Array;
  private readonly mb1: Float32Array;
  private readonly vb1: Float32Array;
  private readonly mW2: Float32Array;
  private readonly vW2: Float32Array;
  private readonly mb2: Float32Array;
  private readonly vb2: Float32Array;

  private dW1!: Float32Array;
  private db1!: Float32Array;
  private dW2!: Float32Array;
  private db2!: Float32Array;

  private Z1!: Float32Array;
  private A1!: Float32Array;
  private P!: Float32Array;
  private lastX: Float32Array = new Float32Array(0);
  private t = 0;

  constructor(
    readonly D0: number,
    readonly D1: number,
    readonly C: number,
    readonly B: number,
    init: CpuMlpInit,
  ) {
    this.W1 = init.W1.slice();
    this.b1 = init.b1.slice();
    this.W2 = init.W2.slice();
    this.b2 = init.b2.slice();
    const z = (n: number) => new Float32Array(n);
    this.mW1 = z(D0 * D1);
    this.vW1 = z(D0 * D1);
    this.mb1 = z(D1);
    this.vb1 = z(D1);
    this.mW2 = z(D1 * C);
    this.vW2 = z(D1 * C);
    this.mb2 = z(C);
    this.vb2 = z(C);
  }

  /** X is [B, D0]. Returns softmax probabilities [B, C]. Caches for backward(). */
  forward(X: Float32Array): Float32Array {
    const { D0, D1, C, B } = this;
    this.Z1 = ref.biasAddCpu(ref.matmulCpu(X, this.W1, B, D0, D1), this.b1, B, D1);
    this.A1 = ref.reluCpu(this.Z1);
    const Z2 = ref.biasAddCpu(ref.matmulCpu(this.A1, this.W2, B, D1, C), this.b2, B, C);
    this.P = ref.softmaxCpu(Z2, B, C);
    this.lastX = X;
    return this.P;
  }

  get probs(): Float32Array {
    return this.P;
  }

  backward(labels: Uint32Array): void {
    const { D0, D1, C, B } = this;
    const dZ2 = ref.softmaxCeBackwardCpu(this.P, labels, B, C); // (p - onehot)/B
    this.dW2 = ref.matmulATBCpu(this.A1, dZ2, B, D1, C);
    this.db2 = ref.biasBackwardCpu(dZ2, B, C);
    const dA1 = ref.matmulABTCpu(dZ2, this.W2, B, C, D1);
    const dZ1 = ref.reluBackwardCpu(dA1, this.Z1);
    this.dW1 = ref.matmulATBCpu(this.lastX, dZ1, B, D0, D1);
    this.db1 = ref.biasBackwardCpu(dZ1, B, D1);
  }

  step(lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    adamUpdate(this.W1, this.dW1, this.mW1, this.vW1, lr, b1, b2, eps, bc1, bc2);
    adamUpdate(this.b1, this.db1, this.mb1, this.vb1, lr, b1, b2, eps, bc1, bc2);
    adamUpdate(this.W2, this.dW2, this.mW2, this.vW2, lr, b1, b2, eps, bc1, bc2);
    adamUpdate(this.b2, this.db2, this.mb2, this.vb2, lr, b1, b2, eps, bc1, bc2);
  }
}

/** Gather a [B, pixels] f32 batch (normalized to [0,1]) from raw uint8 image bytes. */
export function gatherBatchCpu(
  images: Uint8Array,
  indices: Uint32Array,
  pixels: number,
): Float32Array {
  const B = indices.length;
  const out = new Float32Array(B * pixels);
  for (let b = 0; b < B; b++) {
    const src = indices[b] * pixels;
    const dst = b * pixels;
    for (let p = 0; p < pixels; p++) out[dst + p] = images[src + p] / 255;
  }
  return out;
}
