// MNIST loader. Fetches the gzipped IDX files (served same-origin from
// public/mnist/), decompresses them with the browser's DecompressionStream,
// parses the IDX header, and uploads the raw uint8 pixel stream to a GPU storage
// buffer — packed 4 pixels per u32, so all 60k training images fit in ~47 MB.
// A gather shader (see gpu/ops.ts) pulls normalized f32 mini-batches on demand.

const BASE = `${import.meta.env.BASE_URL}mnist/`;

const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
} as const;

export interface MnistSplit {
  images: Uint8Array; // [count * pixels] raw uint8 (kept on CPU for rendering)
  packed: GPUBuffer; // same bytes on the GPU, read as u32 by the gather shader
  labels: Uint8Array; // [count]
  count: number;
}

export interface MnistData {
  train: MnistSplit;
  test: MnistSplit;
  rows: number;
  cols: number;
  pixels: number;
}

export type LoadProgress = (message: string) => void;

async function gunzip(data: ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser lacks DecompressionStream (gzip) support.');
  }
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status} ${res.statusText}`);
  const ab = await res.arrayBuffer();
  // Some static servers send `.gz` files with `Content-Encoding: gzip`, so the
  // browser has already inflated the body. Only decompress if the bytes are
  // still gzip-framed (magic 0x1f 0x8b) — otherwise we'd double-decompress.
  const head = new Uint8Array(ab, 0, Math.min(2, ab.byteLength));
  if (head.length >= 2 && head[0] === 0x1f && head[1] === 0x8b) {
    return await gunzip(ab);
  }
  return ab;
}

interface ImageFile {
  count: number;
  rows: number;
  cols: number;
  data: Uint8Array;
}

function parseImages(buf: ArrayBuffer): ImageFile {
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, false); // IDX is big-endian
  if (magic !== 0x803) throw new Error(`bad image magic 0x${magic.toString(16)} (expected 0x803)`);
  const count = dv.getUint32(4, false);
  const rows = dv.getUint32(8, false);
  const cols = dv.getUint32(12, false);
  const data = new Uint8Array(buf, 16, count * rows * cols);
  return { count, rows, cols, data };
}

function parseLabels(buf: ArrayBuffer): { count: number; data: Uint8Array } {
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, false);
  if (magic !== 0x801) throw new Error(`bad label magic 0x${magic.toString(16)} (expected 0x801)`);
  const count = dv.getUint32(4, false);
  const data = new Uint8Array(buf, 8, count);
  return { count, data };
}

function uploadPacked(device: GPUDevice, bytes: Uint8Array, label: string): GPUBuffer {
  // The GPU buffer must be a multiple of 4 bytes to be read as array<u32>.
  const rem = bytes.byteLength % 4;
  let src = bytes;
  if (rem !== 0) {
    const padded = new Uint8Array(bytes.byteLength + (4 - rem));
    padded.set(bytes);
    src = padded;
  }
  const buf = device.createBuffer({
    label,
    size: src.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, src as Uint8Array<ArrayBuffer>);
  return buf;
}

export async function loadMnist(device: GPUDevice, onProgress?: LoadProgress): Promise<MnistData> {
  onProgress?.('fetching IDX files…');
  const [triBuf, trlBuf, teiBuf, telBuf] = await Promise.all([
    fetchBytes(BASE + FILES.trainImages),
    fetchBytes(BASE + FILES.trainLabels),
    fetchBytes(BASE + FILES.testImages),
    fetchBytes(BASE + FILES.testLabels),
  ]);

  onProgress?.('parsing IDX…');
  const tri = parseImages(triBuf);
  const trl = parseLabels(trlBuf);
  const tei = parseImages(teiBuf);
  const tel = parseLabels(telBuf);
  if (tri.count !== trl.count) throw new Error('train image/label count mismatch');
  if (tei.count !== tel.count) throw new Error('test image/label count mismatch');
  const pixels = tri.rows * tri.cols;

  onProgress?.('uploading to GPU…');
  const train: MnistSplit = {
    images: tri.data,
    packed: uploadPacked(device, tri.data, 'mnist.train.images'),
    labels: trl.data,
    count: tri.count,
  };
  const test: MnistSplit = {
    images: tei.data,
    packed: uploadPacked(device, tei.data, 'mnist.test.images'),
    labels: tel.data,
    count: tei.count,
  };

  return { train, test, rows: tri.rows, cols: tri.cols, pixels };
}

export interface MnistRaw {
  train: { images: Uint8Array; labels: Uint8Array; count: number };
  test: { images: Uint8Array; labels: Uint8Array; count: number };
  rows: number;
  cols: number;
  pixels: number;
}

/** Fetch + parse MNIST to CPU arrays only (no GPU device). Used by the CPU fallback. */
export async function fetchMnistRaw(onProgress?: LoadProgress): Promise<MnistRaw> {
  onProgress?.('fetching IDX files…');
  const [triBuf, trlBuf, teiBuf, telBuf] = await Promise.all([
    fetchBytes(BASE + FILES.trainImages),
    fetchBytes(BASE + FILES.trainLabels),
    fetchBytes(BASE + FILES.testImages),
    fetchBytes(BASE + FILES.testLabels),
  ]);
  onProgress?.('parsing IDX…');
  const tri = parseImages(triBuf);
  const trl = parseLabels(trlBuf);
  const tei = parseImages(teiBuf);
  const tel = parseLabels(telBuf);
  return {
    train: { images: tri.data, labels: trl.data, count: tri.count },
    test: { images: tei.data, labels: tel.data, count: tei.count },
    rows: tri.rows,
    cols: tri.cols,
    pixels: tri.rows * tri.cols,
  };
}
