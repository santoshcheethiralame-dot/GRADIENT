@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = cfg.x;
  let N = cfg.y;
  let c0 = cfg.z;
  let w = cfg.w;
  let idx = gid.x;
  if (idx >= M * w) {
    return;
  }
  let i = idx / w;
  let k = idx % w;
  dst[i * w + k] = src[i * N + c0 + k];
}
