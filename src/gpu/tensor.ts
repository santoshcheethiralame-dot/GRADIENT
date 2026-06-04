// A dense, row-major f32 tensor backed by a single GPUBuffer. This is the unit
// of memory the whole engine operates on: weights, activations, and gradients
// are all GpuTensors allocated once and reused across training steps.

const F32 = 4; // bytes per f32

export interface TensorOptions {
  label?: string;
  /** Extra GPUBufferUsage flags OR'd into the defaults. */
  usage?: number;
}

const DEFAULT_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class GpuTensor {
  readonly device: GPUDevice;
  readonly buffer: GPUBuffer;
  readonly shape: readonly number[];
  /** Element count (product of shape). */
  readonly size: number;
  /** Byte length (size * 4). */
  readonly byteLength: number;
  readonly label: string;
  private destroyed = false;

  constructor(device: GPUDevice, shape: readonly number[], options: TensorOptions = {}) {
    if (shape.length === 0) throw new Error('GpuTensor requires a non-empty shape');
    for (const d of shape) {
      if (!Number.isInteger(d) || d <= 0) {
        throw new Error(`Invalid dimension in shape [${shape.join(', ')}]`);
      }
    }
    this.device = device;
    this.shape = [...shape];
    this.size = shape.reduce((a, b) => a * b, 1);
    this.byteLength = this.size * F32;
    this.label = options.label ?? `tensor[${this.shape.join('×')}]`;
    this.buffer = device.createBuffer({
      label: this.label,
      size: align4(this.byteLength),
      usage: DEFAULT_USAGE | (options.usage ?? 0),
    });
  }

  /** Zero-filled tensor. WebGPU guarantees freshly created buffers are zeroed. */
  static zeros(device: GPUDevice, shape: readonly number[], options?: TensorOptions): GpuTensor {
    return new GpuTensor(device, shape, options);
  }

  /** Tensor initialized from CPU data (length must equal the shape's size). */
  static fromArray(
    device: GPUDevice,
    data: Float32Array,
    shape: readonly number[],
    options?: TensorOptions,
  ): GpuTensor {
    const t = new GpuTensor(device, shape, options);
    if (data.length !== t.size) {
      throw new Error(
        `Data length ${data.length} != shape [${shape.join(', ')}] size ${t.size}`,
      );
    }
    // Cast: TS 5.7's Float32Array is generic over ArrayBufferLike, but
    // writeBuffer requires a non-shared ArrayBuffer-backed view.
    device.queue.writeBuffer(t.buffer, 0, data as Float32Array<ArrayBuffer>);
    return t;
  }

  /** Overwrite contents from a CPU array of matching element count. */
  write(data: Float32Array): void {
    this.assertAlive();
    if (data.length !== this.size) {
      throw new Error(`write(): length ${data.length} != tensor size ${this.size}`);
    }
    this.device.queue.writeBuffer(this.buffer, 0, data as Float32Array<ArrayBuffer>);
  }

  /** Copy the buffer back to the CPU. Submits a copy and awaits the mapping. */
  async toArray(): Promise<Float32Array> {
    this.assertAlive();
    const bytes = align4(this.byteLength);
    const staging = this.device.createBuffer({
      label: `${this.label}:readback`,
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: 'readback' });
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    // getMappedRange() detaches on unmap, so copy out before unmapping.
    const out = new Float32Array(staging.getMappedRange()).slice(0, this.size);
    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy(): void {
    if (!this.destroyed) {
      this.buffer.destroy();
      this.destroyed = true;
    }
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error(`Tensor "${this.label}" has been destroyed`);
  }
}

function align4(bytes: number): number {
  return (bytes + 3) & ~3;
}
