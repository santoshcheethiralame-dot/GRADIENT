// GPU kernel profiler. Measures real amortized GPU time by warming up, then
// submitting many dispatches back-to-back (no readback in between) and dividing
// the wall-clock time once the queue drains via onSubmittedWorkDone(). This
// avoids the readback overhead that pollutes a single round-trip timing and
// works everywhere (no dependency on the optional timestamp-query feature).

import { getGpuContext } from './device';
import { GpuTensor } from './tensor';
import { matmul, createU32Buffer } from './ops';
import { Mlp } from '../nn/mlp';

export interface ProfileRow {
  label: string;
  detail: string;
  ms: number; // GPU time per call
  gflops: number | null; // null for the composite training-step row
}

function randomData(n: number, scale = 1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
  return a;
}

export async function runProfile(): Promise<ProfileRow[]> {
  const { device } = await getGpuContext();
  const rows: ProfileRow[] = [];

  const timeit = async (runOnce: () => void, iters: number): Promise<number> => {
    for (let i = 0; i < 6; i++) runOnce(); // warmup
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) runOnce();
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / iters;
  };

  // matmul across sizes — the core kernel
  const sizes: Array<[number, 'naive' | 'tiled', number]> = [
    [256, 'naive', 120],
    [256, 'tiled', 120],
    [512, 'tiled', 60],
    [1024, 'tiled', 24],
  ];
  for (const [n, variant, iters] of sizes) {
    const a = GpuTensor.fromArray(device, randomData(n * n), [n, n]);
    const b = GpuTensor.fromArray(device, randomData(n * n), [n, n]);
    const out = GpuTensor.zeros(device, [n, n]);
    const ms = await timeit(() => matmul(device, a, b, { variant, out }), iters);
    rows.push({
      label: `matmul · ${variant}`,
      detail: `${n}×${n} · ${n}×${n}`,
      ms,
      gflops: (2 * n * n * n) / (ms / 1000) / 1e9,
    });
    a.destroy();
    b.destroy();
    out.destroy();
  }

  // a full forward + backward training step (no readback)
  const B = 64;
  const D0 = 784;
  const D1 = 128;
  const C = 10;
  const mlp = new Mlp(
    device,
    { inputDim: D0, hiddenDim: D1, outputDim: C, batch: B },
    {
      W1: randomData(D0 * D1, 0.05),
      b1: new Float32Array(D1),
      W2: randomData(D1 * C, 0.1),
      b2: new Float32Array(C),
    },
  );
  const X = GpuTensor.fromArray(device, randomData(B * D0), [B, D0]);
  const labels = new Uint32Array(B);
  for (let i = 0; i < B; i++) labels[i] = i % C;
  const labelBuf = createU32Buffer(device, labels, 'profile.labels');
  const stepMs = await timeit(() => {
    mlp.forward(X);
    mlp.backward(labelBuf);
  }, 60);
  rows.push({
    label: 'train step · fwd+bwd',
    detail: `${D0}→${D1}→${C} · batch ${B}`,
    ms: stepMs,
    gflops: null,
  });
  mlp.destroy();
  X.destroy();

  return rows;
}
