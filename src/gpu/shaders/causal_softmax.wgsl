struct Cfg {
  T: u32,
  scale: f32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.T) {
    return;
  }
  let base = i * cfg.T;
  let lim = i + 1u;

  var m = inp[base] * cfg.scale;
  for (var j = 1u; j < lim; j = j + 1u) {
    m = max(m, inp[base + j] * cfg.scale);
  }

  var s = 0.0;
  for (var j = 0u; j < lim; j = j + 1u) {
    let e = exp(inp[base + j] * cfg.scale - m);
    outp[base + j] = e;
    s = s + e;
  }

  let inv = 1.0 / s;
  for (var j = 0u; j < lim; j = j + 1u) {
    outp[base + j] = outp[base + j] * inv;
  }
  for (var j = lim; j < cfg.T; j = j + 1u) {
    outp[base + j] = 0.0;
  }
}
