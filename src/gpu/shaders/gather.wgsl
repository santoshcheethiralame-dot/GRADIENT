struct Dims {
  batch: u32,
  pixels: u32,
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<storage, read>       packed: array<u32>;
@group(0) @binding(1) var<storage, read>       indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> batch: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = dims.batch * dims.pixels;
  if (i >= total) {
    return;
  }
  let b = i / dims.pixels;
  let p = i % dims.pixels;
  let g = indices[b] * dims.pixels + p;
  let word = packed[g >> 2u];
  let byteVal = (word >> ((g & 3u) * 8u)) & 0xffu;
  batch[i] = f32(byteVal) / 255.0;
}
