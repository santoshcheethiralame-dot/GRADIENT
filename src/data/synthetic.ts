// Synthetic datasets for exercising the training loop before MNIST lands.
// Gaussian blobs: each class is a cloud of points around a random center.
// Well-separated centers make this reliably learnable — ideal for a convergence
// self-test that must pass on every load.

export interface Dataset {
  X: Float32Array; // [samples * dim], row-major
  labels: Uint32Array; // [samples]
  samples: number;
  dim: number;
  classes: number;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal sample via Box-Muller. */
export function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export interface BlobOptions {
  seed?: number;
  spread?: number; // per-axis stddev of each cluster
  separation?: number; // scale of the random class centers
}

export function makeBlobs(
  samples: number,
  dim: number,
  classes: number,
  options: BlobOptions = {},
): Dataset {
  const { seed = 1, spread = 0.6, separation = 3 } = options;
  const rng = mulberry32(seed);

  const centers: Float32Array[] = [];
  for (let c = 0; c < classes; c++) {
    const ctr = new Float32Array(dim);
    for (let d = 0; d < dim; d++) ctr[d] = (rng() * 2 - 1) * separation;
    centers.push(ctr);
  }

  const X = new Float32Array(samples * dim);
  const labels = new Uint32Array(samples);
  for (let s = 0; s < samples; s++) {
    const c = Math.floor(rng() * classes);
    labels[s] = c;
    const ctr = centers[c];
    for (let d = 0; d < dim; d++) {
      X[s * dim + d] = ctr[d] + gaussian(rng) * spread;
    }
  }
  return { X, labels, samples, dim, classes };
}
