const TILE: u32 = 16u;

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

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let col = gid.x;
  let row = gid.y;
  let lx = lid.x;
  let ly = lid.y;

  var acc = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * TILE + lx;
    if (row < dims.M && aCol < dims.K) {
      tileA[ly * TILE + lx] = a[row * dims.K + aCol];
    } else {
      tileA[ly * TILE + lx] = 0.0;
    }

    let bRow = t * TILE + ly;
    if (bRow < dims.K && col < dims.N) {
      tileB[ly * TILE + lx] = b[bRow * dims.N + col];
    } else {
      tileB[ly * TILE + lx] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[ly * TILE + k] * tileB[k * TILE + lx];
    }

    workgroupBarrier();
  }

  if (row < dims.M && col < dims.N) {
    c[row * dims.N + col] = acc;
  }
}
