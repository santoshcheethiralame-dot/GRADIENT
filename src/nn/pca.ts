// Tiny PCA → 2-D projection (top-2 principal components via power iteration).
// Runs on the CPU over a small [n × d] activation matrix already read back from
// the GPU — used to project hidden-layer activations for the embedding scatter.

function powerIteration(cov: Float32Array, d: number, deflate: Float32Array | null): Float32Array {
  const v = new Float32Array(d);
  let seed = 1234567;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff - 0.5;
  };
  for (let i = 0; i < d; i++) v[i] = rnd();

  for (let it = 0; it < 64; it++) {
    const w = new Float32Array(d);
    for (let a = 0; a < d; a++) {
      let s = 0;
      const row = a * d;
      for (let b = 0; b < d; b++) s += cov[row + b] * v[b];
      w[a] = s;
    }
    // deflate against an already-found eigenvector to get the next one
    if (deflate) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += w[j] * deflate[j];
      for (let j = 0; j < d; j++) w[j] -= dot * deflate[j];
    }
    let norm = 0;
    for (let j = 0; j < d; j++) norm += w[j] * w[j];
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < d; j++) v[j] = w[j] / norm;
  }
  // stabilize sign (largest-magnitude component positive) so the projection
  // doesn't arbitrarily mirror between updates
  let mi = 0;
  for (let j = 1; j < d; j++) if (Math.abs(v[j]) > Math.abs(v[mi])) mi = j;
  if (v[mi] < 0) for (let j = 0; j < d; j++) v[j] = -v[j];
  return v;
}

/** Project [n × d] data onto its top-2 principal components → [n × 2], scaled to ~[-1,1]. */
export function pca2d(data: Float32Array, n: number, d: number): Float32Array {
  const out = new Float32Array(n * 2);
  if (n === 0 || d === 0) return out;

  const mean = new Float32Array(d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let j = 0; j < d; j++) mean[j] += data[o + j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  const X = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let j = 0; j < d; j++) X[o + j] = data[o + j] - mean[j];
  }

  // covariance C = Xᵀ X / n (symmetric)
  const cov = new Float32Array(d * d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    for (let a = 0; a < d; a++) {
      const xa = X[o + a];
      if (xa === 0) continue;
      const row = a * d;
      for (let b = a; b < d; b++) cov[row + b] += xa * X[o + b];
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      const v = cov[a * d + b] / n;
      cov[a * d + b] = v;
      cov[b * d + a] = v;
    }
  }

  const v1 = powerIteration(cov, d, null);
  const v2 = powerIteration(cov, d, v1);

  let maxAbs = 1e-9;
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let p1 = 0;
    let p2 = 0;
    for (let j = 0; j < d; j++) {
      p1 += X[o + j] * v1[j];
      p2 += X[o + j] * v2[j];
    }
    out[i * 2] = p1;
    out[i * 2 + 1] = p2;
    maxAbs = Math.max(maxAbs, Math.abs(p1), Math.abs(p2));
  }
  for (let i = 0; i < out.length; i++) out[i] /= maxAbs;
  return out;
}
