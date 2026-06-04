// ReLU backward: dX[i] = dOut[i] · (fwdIn[i] > 0 ? 1 : 0).
// `fwd` is the forward INPUT to ReLU (the pre-activation). The mask is the
// subgradient of max(0,x): 0 at x <= 0, else 1.

struct Dims {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read>       dOut: array<f32>;
@group(0) @binding(1) var<storage, read>       fwd: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  let mask = select(0.0, 1.0, fwd[i] > 0.0);
  dX[i] = dOut[i] * mask;
}
