// SGD parameter update (in-place): w -= lr · g. The simplest optimizer — one
// elementwise pass over a parameter tensor and its gradient.

struct Sgd {
  n: u32,
  lr: f32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>       g: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Sgd;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.n) {
    return;
  }
  w[i] = w[i] - cfg.lr * g[i];
}
