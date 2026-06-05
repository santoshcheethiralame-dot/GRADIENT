struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.M) {
    return;
  }
  let base = row * dims.N;

  var m = inp[base];
  for (var j = 1u; j < dims.N; j = j + 1u) {
    m = max(m, inp[base + j]);
  }

  var s = 0.0;
  for (var j = 0u; j < dims.N; j = j + 1u) {
    let e = exp(inp[base + j] - m);
    outp[base + j] = e;
    s = s + e;
  }

  let inv = 1.0 / s;
  for (var j = 0u; j < dims.N; j = j + 1u) {
    outp[base + j] = outp[base + j] * inv;
  }
}
