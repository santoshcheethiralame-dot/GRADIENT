// Adam parameter update (in-place). Maintains per-parameter first/second moment
// estimates m, v across steps:
//   m = β1·m + (1-β1)·g
//   v = β2·v + (1-β2)·g²
//   ŵ -= lr · (m / bc1) / (sqrt(v / bc2) + eps)
// where bc1 = 1-β1^t and bc2 = 1-β2^t are the bias-correction terms, computed
// on the host each step and passed in (saves a pow() in the shader).

struct Adam {
  n: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  bc1: f32, // 1 - beta1^t
  bc2: f32, // 1 - beta2^t
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>       g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform>             cfg: Adam;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.n) {
    return;
  }
  let gi = g[i];
  let mi = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * gi;
  let vi = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  let mhat = mi / cfg.bc1;
  let vhat = vi / cfg.bc2;
  w[i] = w[i] - cfg.lr * mhat / (sqrt(vhat) + cfg.eps);
}
