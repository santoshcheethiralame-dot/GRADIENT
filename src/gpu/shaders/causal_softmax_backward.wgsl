struct Cfg {
  T: u32,
  scale: f32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       attn: array<f32>;
@group(0) @binding(1) var<storage, read>       dattn: array<f32>;
@group(0) @binding(2) var<storage, read_write> dscores: array<f32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.T) {
    return;
  }
  let base = i * cfg.T;
  let lim = i + 1u;

  var sumd = 0.0;
  for (var j = 0u; j < lim; j = j + 1u) {
    sumd = sumd + attn[base + j] * dattn[base + j];
  }
  for (var j = 0u; j < lim; j = j + 1u) {
    dscores[base + j] = cfg.scale * attn[base + j] * (dattn[base + j] - sumd);
  }
  for (var j = lim; j < cfg.T; j = j + 1u) {
    dscores[base + j] = 0.0;
  }
}
