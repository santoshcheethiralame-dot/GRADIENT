import type { GpuTensor } from '../gpu/tensor';
import type { Mlp } from './mlp';

export interface GradCheckResult {
  maxRelErr: number;
  maxAbsErr: number;
}

export async function numericalGradCheck(
  device: GPUDevice,
  mlp: Mlp,
  X: GpuTensor,
  labels: GPUBuffer,
  param: GpuTensor,
  analyticGrad: GpuTensor,
  eps = 1e-2,
): Promise<GradCheckResult> {
  const analytic = await analyticGrad.toArray();
  const original = await param.toArray();
  const n = param.size;
  const scratch = new Float32Array(1);

  let maxRel = 0;
  let maxAbs = 0;

  for (let i = 0; i < n; i++) {
    const w = original[i];

    scratch[0] = w + eps;
    device.queue.writeBuffer(param.buffer, i * 4, scratch);
    const lPlus = await mlp.forwardLoss(X, labels);

    scratch[0] = w - eps;
    device.queue.writeBuffer(param.buffer, i * 4, scratch);
    const lMinus = await mlp.forwardLoss(X, labels);

    scratch[0] = w;
    device.queue.writeBuffer(param.buffer, i * 4, scratch);

    const numerical = (lPlus - lMinus) / (2 * eps);
    const a = analytic[i];
    const abs = Math.abs(numerical - a);
    const denom = Math.max(Math.abs(numerical), Math.abs(a), 1e-3);
    const rel = abs / denom;

    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
  }

  return { maxRelErr: maxRel, maxAbsErr: maxAbs };
}
