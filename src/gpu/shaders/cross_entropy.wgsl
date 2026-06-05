struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       probs: array<f32>;
@group(0) @binding(1) var<storage, read>       labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.M) {
    return;
  }
  let label = labels[row];
  let p = probs[row * dims.N + label];
  losses[row] = -log(max(p, 1e-12));
}
