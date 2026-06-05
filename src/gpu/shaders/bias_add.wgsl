struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read>       bias: array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = dims.M * dims.N;
  if (idx >= total) {
    return;
  }
  let col = idx % dims.N;
  y[idx] = y[idx] + bias[col];
}
