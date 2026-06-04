// Sigmoid (out-of-place): out[i] = 1 / (1 + exp(-in[i])).
// Saturates cleanly without NaN: very negative in -> exp(+large) -> 0; very
// positive in -> exp(0-) -> 1.

struct Dims {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  outp[i] = 1.0 / (1.0 + exp(-inp[i]));
}
