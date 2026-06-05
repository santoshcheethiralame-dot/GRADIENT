export function matmulCpu(
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
): Float32Array {
  const c = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        acc += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = acc;
    }
  }
  return c;
}

export function biasAddCpu(
  y: Float32Array,
  bias: Float32Array,
  M: number,
  N: number,
): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      out[i * N + j] = y[i * N + j] + bias[j];
    }
  }
  return out;
}

export function reluCpu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(0, x[i]);
  return out;
}

export function sigmoidCpu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
  return out;
}

export function softmaxCpu(x: Float32Array, M: number, N: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    const base = i * N;
    let m = -Infinity;
    for (let j = 0; j < N; j++) m = Math.max(m, x[base + j]);
    let s = 0;
    for (let j = 0; j < N; j++) {
      const e = Math.exp(x[base + j] - m);
      out[base + j] = e;
      s += e;
    }
    const inv = 1 / s;
    for (let j = 0; j < N; j++) out[base + j] *= inv;
  }
  return out;
}

export function crossEntropyCpu(
  probs: Float32Array,
  labels: Uint32Array,
  M: number,
  N: number,
): Float32Array {
  const out = new Float32Array(M);
  for (let i = 0; i < M; i++) {
    const p = probs[i * N + labels[i]];
    out[i] = -Math.log(Math.max(p, 1e-12));
  }
  return out;
}

export function matmulATBCpu(
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
): Float32Array {
  const c = new Float32Array(K * N);
  for (let k = 0; k < K; k++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let m = 0; m < M; m++) acc += a[m * K + k] * b[m * N + n];
      c[k * N + n] = acc;
    }
  }
  return c;
}

export function matmulABTCpu(
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
): Float32Array {
  const c = new Float32Array(M * K);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      let acc = 0;
      for (let n = 0; n < N; n++) acc += a[m * N + n] * b[k * N + n];
      c[m * K + k] = acc;
    }
  }
  return c;
}

export function biasBackwardCpu(dY: Float32Array, M: number, N: number): Float32Array {
  const db = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let acc = 0;
    for (let m = 0; m < M; m++) acc += dY[m * N + n];
    db[n] = acc;
  }
  return db;
}

export function reluBackwardCpu(dOut: Float32Array, fwd: Float32Array): Float32Array {
  const out = new Float32Array(dOut.length);
  for (let i = 0; i < dOut.length; i++) out[i] = fwd[i] > 0 ? dOut[i] : 0;
  return out;
}

export function sigmoidBackwardCpu(dOut: Float32Array, y: Float32Array): Float32Array {
  const out = new Float32Array(dOut.length);
  for (let i = 0; i < dOut.length; i++) out[i] = dOut[i] * y[i] * (1 - y[i]);
  return out;
}

export function softmaxCeBackwardCpu(
  probs: Float32Array,
  labels: Uint32Array,
  M: number,
  N: number,
): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const onehot = j === labels[i] ? 1 : 0;
      out[i * N + j] = (probs[i * N + j] - onehot) / M;
    }
  }
  return out;
}

export function sgdStepCpu(w: Float32Array, g: Float32Array, lr: number): Float32Array {
  const out = new Float32Array(w.length);
  for (let i = 0; i < w.length; i++) out[i] = w[i] - lr * g[i];
  return out;
}

export interface AdamCpuConfig {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export function adamCpu(w0: Float32Array, grads: Float32Array[], cfg: AdamCpuConfig): Float32Array {
  const n = w0.length;
  const w = w0.slice();
  const m = new Float32Array(n);
  const v = new Float32Array(n);
  const { lr, beta1, beta2, eps } = cfg;
  for (let t = 1; t <= grads.length; t++) {
    const g = grads[t - 1];
    const bc1 = 1 - beta1 ** t;
    const bc2 = 1 - beta2 ** t;
    for (let i = 0; i < n; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
      const mhat = m[i] / bc1;
      const vhat = v[i] / bc2;
      w[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }
  return w;
}

export function accuracy(probs: Float32Array, labels: Uint32Array, M: number, N: number): number {
  let correct = 0;
  for (let i = 0; i < M; i++) {
    let best = -Infinity;
    let arg = 0;
    for (let j = 0; j < N; j++) {
      const p = probs[i * N + j];
      if (p > best) {
        best = p;
        arg = j;
      }
    }
    if (arg === labels[i]) correct++;
  }
  return correct / M;
}

export interface ErrorMetrics {
  maxAbsErr: number;
  maxRelErr: number;
}

export function compareArrays(got: Float32Array, ref: Float32Array): ErrorMetrics {
  if (got.length !== ref.length) {
    throw new Error(`length mismatch: got ${got.length}, ref ${ref.length}`);
  }
  let maxAbs = 0;
  let maxRef = 0;
  for (let i = 0; i < ref.length; i++) {
    const abs = Math.abs(got[i] - ref[i]);
    if (abs > maxAbs) maxAbs = abs;
    const r = Math.abs(ref[i]);
    if (r > maxRef) maxRef = r;
  }
  return { maxAbsErr: maxAbs, maxRelErr: maxRef > 0 ? maxAbs / maxRef : maxAbs };
}
