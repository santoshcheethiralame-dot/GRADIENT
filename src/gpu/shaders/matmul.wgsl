struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  if (row >= dims.M || col >= dims.N) {
    return;
  }

  var acc = 0.0;
  for (var k = 0u; k < dims.K; k = k + 1u) {
    acc = acc + a[row * dims.K + k] * b[k * dims.N + col];
  }
  c[row * dims.N + col] = acc;
}
