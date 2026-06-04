// A 2-layer MLP (one hidden ReLU layer, softmax output) whose forward and
// backward passes run entirely on the GPU. All tensors — params, activations,
// gradients — are allocated once for a fixed batch size and reused every step
// (the "allocate once, reuse" pattern). Phase 4 adds an optimizer on top.

import { GpuTensor } from '../gpu/tensor';
import {
  matmul,
  biasAdd,
  relu,
  softmax,
  crossEntropy,
  matmulATB,
  matmulABT,
  biasBackward,
  reluBackward,
  softmaxCeBackward,
} from '../gpu/ops';

export interface MlpDims {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  batch: number;
}

export interface MlpInit {
  W1: Float32Array;
  b1: Float32Array;
  W2: Float32Array;
  b2: Float32Array;
}

export class Mlp {
  readonly device: GPUDevice;
  readonly dims: MlpDims;

  // parameters
  readonly W1: GpuTensor;
  readonly b1: GpuTensor;
  readonly W2: GpuTensor;
  readonly b2: GpuTensor;

  // parameter gradients (+ input gradient)
  readonly dW1: GpuTensor;
  readonly db1: GpuTensor;
  readonly dW2: GpuTensor;
  readonly db2: GpuTensor;
  readonly dX: GpuTensor;

  // forward-pass caches
  readonly Z1: GpuTensor; // pre-activation (hidden)
  readonly A1: GpuTensor; // ReLU activation
  readonly Z2: GpuTensor; // logits
  readonly P: GpuTensor; // softmax probabilities
  readonly losses: GpuTensor; // per-sample CE

  // backward intermediates
  readonly dZ2: GpuTensor;
  readonly dA1: GpuTensor;
  readonly dZ1: GpuTensor;

  private lastX: GpuTensor | null = null;
  private owned: GpuTensor[];

  constructor(device: GPUDevice, dims: MlpDims, init?: MlpInit) {
    const { inputDim: D0, hiddenDim: D1, outputDim: C, batch: B } = dims;
    this.device = device;
    this.dims = dims;

    const param = (data: Float32Array | undefined, shape: number[], label: string) =>
      data
        ? GpuTensor.fromArray(device, data, shape, { label })
        : GpuTensor.zeros(device, shape, { label });

    this.W1 = param(init?.W1, [D0, D1], 'W1');
    this.b1 = param(init?.b1, [D1], 'b1');
    this.W2 = param(init?.W2, [D1, C], 'W2');
    this.b2 = param(init?.b2, [C], 'b2');

    this.dW1 = GpuTensor.zeros(device, [D0, D1], { label: 'dW1' });
    this.db1 = GpuTensor.zeros(device, [D1], { label: 'db1' });
    this.dW2 = GpuTensor.zeros(device, [D1, C], { label: 'dW2' });
    this.db2 = GpuTensor.zeros(device, [C], { label: 'db2' });
    this.dX = GpuTensor.zeros(device, [B, D0], { label: 'dX' });

    this.Z1 = GpuTensor.zeros(device, [B, D1], { label: 'Z1' });
    this.A1 = GpuTensor.zeros(device, [B, D1], { label: 'A1' });
    this.Z2 = GpuTensor.zeros(device, [B, C], { label: 'Z2' });
    this.P = GpuTensor.zeros(device, [B, C], { label: 'P' });
    this.losses = GpuTensor.zeros(device, [B], { label: 'losses' });

    this.dZ2 = GpuTensor.zeros(device, [B, C], { label: 'dZ2' });
    this.dA1 = GpuTensor.zeros(device, [B, D1], { label: 'dA1' });
    this.dZ1 = GpuTensor.zeros(device, [B, D1], { label: 'dZ1' });

    this.owned = [
      this.W1, this.b1, this.W2, this.b2,
      this.dW1, this.db1, this.dW2, this.db2, this.dX,
      this.Z1, this.A1, this.Z2, this.P, this.losses,
      this.dZ2, this.dA1, this.dZ1,
    ];
  }

  /** X[B,D0] -> probs[B,C]. Caches Z1, A1, Z2, P for the backward pass. */
  forward(X: GpuTensor): GpuTensor {
    const d = this.device;
    matmul(d, X, this.W1, { out: this.Z1 }); // Z1 = X @ W1
    biasAdd(d, this.Z1, this.b1); //            Z1 += b1
    relu(d, this.Z1, this.A1); //               A1 = relu(Z1)
    matmul(d, this.A1, this.W2, { out: this.Z2 }); // Z2 = A1 @ W2
    biasAdd(d, this.Z2, this.b2); //            Z2 += b2
    softmax(d, this.Z2, this.P); //             P = softmax(Z2)
    this.lastX = X;
    return this.P;
  }

  /** Forward + cross-entropy; returns the mean loss over the batch (one readback). */
  async forwardLoss(X: GpuTensor, labels: GPUBuffer): Promise<number> {
    this.forward(X);
    crossEntropy(this.device, this.P, labels, this.losses);
    const l = await this.losses.toArray();
    let s = 0;
    for (let i = 0; i < l.length; i++) s += l[i];
    return s / l.length;
  }

  /** Backprop from the cached forward state. Fills dW1, db1, dW2, db2, dX. */
  backward(labels: GPUBuffer): void {
    if (!this.lastX) throw new Error('backward() called before forward()');
    const d = this.device;
    softmaxCeBackward(d, this.P, labels, this.dZ2); // dZ2 = (P - onehot) / B
    matmulATB(d, this.A1, this.dZ2, this.dW2); //     dW2 = A1ᵀ @ dZ2
    biasBackward(d, this.dZ2, this.db2); //           db2 = Σ_b dZ2
    matmulABT(d, this.dZ2, this.W2, this.dA1); //     dA1 = dZ2 @ W2ᵀ
    reluBackward(d, this.dA1, this.Z1, this.dZ1); //  dZ1 = dA1 ⊙ (Z1 > 0)
    matmulATB(d, this.lastX, this.dZ1, this.dW1); //  dW1 = Xᵀ @ dZ1
    biasBackward(d, this.dZ1, this.db1); //           db1 = Σ_b dZ1
    matmulABT(d, this.dZ1, this.W1, this.dX); //      dX  = dZ1 @ W1ᵀ
  }

  /** Learnable parameters, in a fixed order. */
  params(): GpuTensor[] {
    return [this.W1, this.b1, this.W2, this.b2];
  }

  /** Gradients, in the same order as params(). */
  grads(): GpuTensor[] {
    return [this.dW1, this.db1, this.dW2, this.db2];
  }

  destroy(): void {
    for (const t of this.owned) t.destroy();
  }
}
