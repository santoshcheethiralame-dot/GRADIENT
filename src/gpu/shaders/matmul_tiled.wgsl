// Tiled dense matmul: C[M,N] = A[M,K] @ B[K,N]
//
// Each workgroup computes one TILE×TILE block of C. The workgroup cooperatively
// stages a TILE×TILE block of A and of B into workgroup-shared memory, does the
// partial dot products from that fast memory, then advances to the next tile
// along K. This turns O(N^3) global loads into O(N^3 / TILE) — a ~TILE-fold cut
// in global-memory traffic, the classic GPU matmul optimization.
//
// Boundary handling: invocations whose (row,col) fall outside C still execute
// (loading zeros) so that every invocation reaches the workgroupBarriers — the
// barriers require uniform control flow across the workgroup. Only the final
// store is guarded.

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

var<workgroup> tileA: array<f32, 256>; // TILE * TILE
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let col = gid.x; // along N
  let row = gid.y; // along M
  let lx = lid.x;
  let ly = lid.y;

  var acc = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t = t + 1u) {
    // Stage A[row, t*TILE + lx] into tileA[ly][lx].
    let aCol = t * TILE + lx;
    if (row < dims.M && aCol < dims.K) {
      tileA[ly * TILE + lx] = a[row * dims.K + aCol];
    } else {
      tileA[ly * TILE + lx] = 0.0;
    }

    // Stage B[t*TILE + ly, col] into tileB[ly][lx].
    let bRow = t * TILE + ly;
    if (bRow < dims.K && col < dims.N) {
      tileB[ly * TILE + lx] = b[bRow * dims.N + col];
    } else {
      tileB[ly * TILE + lx] = 0.0;
    }

    workgroupBarrier(); // all stages visible before consuming

    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[ly * TILE + k] * tileB[k * TILE + lx];
    }

    workgroupBarrier(); // done consuming before overwriting next tile
  }

  if (row < dims.M && col < dims.N) {
    c[row * dims.N + col] = acc;
  }
}
