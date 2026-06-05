// Row-wise layer norm with affine params:
//   out[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]
// One invocation per row, looping over the N features. eps = 1e-5 to match the
// CPU reference (layerNormRows in nanogpt.ts).

struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.M) {
    return;
  }
  let base = row * dims.N;
  let nf = f32(dims.N);

  var mean = 0.0;
  for (var j = 0u; j < dims.N; j = j + 1u) {
    mean = mean + inp[base + j];
  }
  mean = mean / nf;

  var varc = 0.0;
  for (var j = 0u; j < dims.N; j = j + 1u) {
    let d = inp[base + j] - mean;
    varc = varc + d * d;
  }
  varc = varc / nf;

  let inv = 1.0 / sqrt(varc + 1e-5);
  for (var j = 0u; j < dims.N; j = j + 1u) {
    outp[base + j] = (inp[base + j] - mean) * inv * gamma[j] + beta[j];
  }
}
