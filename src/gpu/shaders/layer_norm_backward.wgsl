struct Dims {
  M: u32,
  N: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(5) var<uniform>             dims: Dims;

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

  var sumDxhat = 0.0;
  var sumDxhatXhat = 0.0;
  for (var j = 0u; j < dims.N; j = j + 1u) {
    let xh = (inp[base + j] - mean) * inv;
    let dxh = dy[base + j] * gamma[j];
    sumDxhat = sumDxhat + dxh;
    sumDxhatXhat = sumDxhatXhat + dxh * xh;
  }

  for (var j = 0u; j < dims.N; j = j + 1u) {
    let xh = (inp[base + j] - mean) * inv;
    xhat[base + j] = xh;
    dx[base + j] = inv * (dy[base + j] * gamma[j] - sumDxhat / nf - xh * (sumDxhatXhat / nf));
  }
}
