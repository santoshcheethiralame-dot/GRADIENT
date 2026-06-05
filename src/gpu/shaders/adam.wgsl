struct Adam {
  n: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  bc1: f32,
  bc2: f32,
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
