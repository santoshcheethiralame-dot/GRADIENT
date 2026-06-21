struct Dims {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read>       dout: array<f32>;
@group(0) @binding(1) var<storage, read>       pre: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  let x = pre[i];
  let c = 0.7978845608028654;
  let u = c * (x + 0.044715 * x * x * x);
  let th = tanh(u);
  let du = c * (1.0 + 0.134145 * x * x);
  let g = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
  outp[i] = dout[i] * g;
}
