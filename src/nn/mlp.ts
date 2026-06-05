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

  readonly W1: GpuTensor;
  readonly b1: GpuTensor;
  readonly W2: GpuTensor;
  readonly b2: GpuTensor;

  readonly dW1: GpuTensor;
  readonly db1: GpuTensor;
  readonly dW2: GpuTensor;
  readonly db2: GpuTensor;
  readonly dX: GpuTensor;

  readonly Z1: GpuTensor;
  readonly A1: GpuTensor;
  readonly Z2: GpuTensor;
  readonly P: GpuTensor;
  readonly losses: GpuTensor;

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

  forward(X: GpuTensor): GpuTensor {
    const d = this.device;
    matmul(d, X, this.W1, { out: this.Z1 });
    biasAdd(d, this.Z1, this.b1);
    relu(d, this.Z1, this.A1);
    matmul(d, this.A1, this.W2, { out: this.Z2 });
    biasAdd(d, this.Z2, this.b2);
    softmax(d, this.Z2, this.P);
    this.lastX = X;
    return this.P;
  }

  async forwardLoss(X: GpuTensor, labels: GPUBuffer): Promise<number> {
    this.forward(X);
    crossEntropy(this.device, this.P, labels, this.losses);
    const l = await this.losses.toArray();
    let s = 0;
    for (let i = 0; i < l.length; i++) s += l[i];
    return s / l.length;
  }

  backward(labels: GPUBuffer): void {
    if (!this.lastX) throw new Error('backward() called before forward()');
    const d = this.device;
    softmaxCeBackward(d, this.P, labels, this.dZ2);
    matmulATB(d, this.A1, this.dZ2, this.dW2);
    biasBackward(d, this.dZ2, this.db2);
    matmulABT(d, this.dZ2, this.W2, this.dA1);
    reluBackward(d, this.dA1, this.Z1, this.dZ1);
    matmulATB(d, this.lastX, this.dZ1, this.dW1);
    biasBackward(d, this.dZ1, this.db1);
    matmulABT(d, this.dZ1, this.W1, this.dX);
  }

  params(): GpuTensor[] {
    return [this.W1, this.b1, this.W2, this.b2];
  }

  grads(): GpuTensor[] {
    return [this.dW1, this.db1, this.dW2, this.db2];
  }

  destroy(): void {
    for (const t of this.owned) t.destroy();
  }
}
