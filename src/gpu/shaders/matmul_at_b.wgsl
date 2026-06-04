// Transposed-left matmul: C[K,N] = Aᵀ @ B, where A is [M,K] and B is [M,N].
//   C[k,n] = Σ_m A[m,k] · B[m,n]
// This is the weight gradient dW = Xᵀ @ dY (X is [batch,in], dY is [batch,out]).
// Naive (no shared-memory tiling) — correctness first; can be tiled later.

struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _p: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>; // [M,K]
@group(0) @binding(1) var<storage, read>       b: array<f32>; // [M,N]
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [K,N]
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x; // along N
  let k = gid.y; // along K
  if (k >= dims.K || n >= dims.N) {
    return;
  }
  var acc = 0.0;
  for (var m = 0u; m < dims.M; m = m + 1u) {
    acc = acc + a[m * dims.K + k] * b[m * dims.N + n];
  }
  c[k * dims.N + n] = acc;
}
