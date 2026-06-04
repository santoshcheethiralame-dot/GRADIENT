// Numerical gradient checking — the only reliable way to trust GPU backprop.
// For each element w of a parameter we perturb it on the GPU by ±ε, re-run the
// forward pass to get the loss, and form the central-difference estimate
//   ∂L/∂w ≈ (L(w+ε) - L(w-ε)) / 2ε
// then compare to the analytic gradient produced by Mlp.backward(). They should
// agree to a small relative error if (and only if) the backward shaders are
// correct. The forward loss is read back from the actual GPU pipeline, so this
// is an end-to-end check of the real kernels, not a CPU re-derivation.

import type { GpuTensor } from '../gpu/tensor';
import type { Mlp } from './mlp';

export interface GradCheckResult {
  maxRelErr: number;
  maxAbsErr: number;
}

/**
 * Compare the analytic gradient of `param` (already in `analyticGrad`, computed
 * by a prior backward()) against a numerical estimate obtained by perturbing
 * `param` on the GPU and re-running the forward loss.
 *
 * Assumes mlp.forward()+backward() were already run for (X, labels) so that
 * `analyticGrad` holds the analytic values. Restores every perturbed element.
 */
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

    scratch[0] = w; // restore
    device.queue.writeBuffer(param.buffer, i * 4, scratch);

    const numerical = (lPlus - lMinus) / (2 * eps);
    const a = analytic[i];
    const abs = Math.abs(numerical - a);
    // Floor the denominator so near-zero gradients (tiny absolute error) don't
    // produce a misleadingly huge relative error.
    const denom = Math.max(Math.abs(numerical), Math.abs(a), 1e-3);
    const rel = abs / denom;

    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
  }

  return { maxRelErr: maxRel, maxAbsErr: maxAbs };
}
