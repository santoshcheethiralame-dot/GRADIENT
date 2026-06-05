struct Dims {
  M: u32,
  N: u32,
  K: u32,
  _p: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  let m = gid.y;
  if (m >= dims.M || k >= dims.K) {
    return;
  }
  var acc = 0.0;
  for (var n = 0u; n < dims.N; n = n + 1u) {
    acc = acc + a[m * dims.N + n] * b[k * dims.N + n];
  }
  c[m * dims.K + k] = acc;
}
