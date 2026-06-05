import { GpuTensor } from '../gpu/tensor';
import { sgdStep, adamStep } from '../gpu/ops';

export interface Optimizer {
  step(grads: GpuTensor[]): void;
}

export class SGD implements Optimizer {
  constructor(
    private readonly device: GPUDevice,
    private readonly params: GpuTensor[],
    public lr: number,
  ) {}

  step(grads: GpuTensor[]): void {
    for (let i = 0; i < this.params.length; i++) {
      sgdStep(this.device, this.params[i], grads[i], this.lr);
    }
  }
}

export interface AdamConfig {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

export class Adam implements Optimizer {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  private readonly m: GpuTensor[];
  private readonly v: GpuTensor[];
  private t = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly params: GpuTensor[],
    cfg: AdamConfig = {},
  ) {
    this.lr = cfg.lr ?? 1e-3;
    this.beta1 = cfg.beta1 ?? 0.9;
    this.beta2 = cfg.beta2 ?? 0.999;
    this.eps = cfg.eps ?? 1e-8;
    this.m = params.map((p, i) => GpuTensor.zeros(device, p.shape, { label: `adam.m${i}` }));
    this.v = params.map((p, i) => GpuTensor.zeros(device, p.shape, { label: `adam.v${i}` }));
  }

  get step_count(): number {
    return this.t;
  }

  step(grads: GpuTensor[]): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      adamStep(this.device, this.params[i], grads[i], this.m[i], this.v[i], {
        lr: this.lr,
        beta1: this.beta1,
        beta2: this.beta2,
        eps: this.eps,
        bc1,
        bc2,
      });
    }
  }

  destroy(): void {
    for (const t of [...this.m, ...this.v]) t.destroy();
  }
}
