// Bias gradient: db[n] = Σ_m dY[m,n] — reduce the upstream gradient over the
// batch dimension. One invocation per output feature n.

struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       dY: array<f32>; // [M,N]
@group(0) @binding(1) var<storage, read_write> db: array<f32>; // [N]
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= dims.N) {
    return;
  }
  var acc = 0.0;
  for (var m = 0u; m < dims.M; m = m + 1u) {
    acc = acc + dY[m * dims.N + n];
  }
  db[n] = acc;
}
