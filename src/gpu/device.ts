// WebGPU device acquisition and lifecycle. A single shared GPUDevice is created
// lazily and reused everywhere — there is exactly one GPUQueue per device in the
// current spec, and all our submissions are ordered through it.

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: GPUAdapterInfo;
  /** Best-effort human-readable adapter description. */
  label: string;
}

export class WebGpuUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGpuUnsupportedError';
  }
}

let contextPromise: Promise<GpuContext> | null = null;

/** True if the WebGPU API surface exists at all (does not guarantee an adapter). */
export function isWebGpuSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && !!navigator.gpu;
}

/** Lazily initialize (once) and return the shared WebGPU context. */
export function getGpuContext(): Promise<GpuContext> {
  if (!contextPromise) {
    contextPromise = initGpu().catch((err) => {
      contextPromise = null; // allow a later retry
      throw err;
    });
  }
  return contextPromise;
}

async function initGpu(): Promise<GpuContext> {
  if (!isWebGpuSupported()) {
    throw new WebGpuUnsupportedError(
      'WebGPU is not available in this browser. Use Chrome or Edge 113+, a recent Safari, ' +
        'or enable the WebGPU flag in Firefox.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnsupportedError(
      'No compatible GPU adapter was found. Your hardware or graphics driver may not support WebGPU.',
    );
  }

  const device = await adapter.requestDevice({ label: 'gradient-device' });

  // Surface shader/programming errors that are not tied to a specific await
  // (e.g. out-of-bounds writes). Invaluable while authoring compute shaders.
  device.addEventListener('uncapturederror', (event) => {
    const err = (event as GPUUncapturedErrorEvent).error;
    console.error('[WebGPU uncaptured error]', (err as GPUError).message || err);
  });

  void device.lost.then((lost) => {
    if (lost.reason !== 'destroyed') {
      console.error(`[WebGPU device lost] ${lost.reason}: ${lost.message}`);
      contextPromise = null;
    }
  });

  const info = await resolveAdapterInfo(adapter);
  return { adapter, device, info, label: formatAdapterLabel(info) };
}

async function resolveAdapterInfo(adapter: GPUAdapter): Promise<GPUAdapterInfo> {
  // Current spec exposes `adapter.info` synchronously; older implementations
  // expose an async `requestAdapterInfo()`.
  const direct = (adapter as { info?: GPUAdapterInfo }).info;
  if (direct) return direct;
  const legacy = (adapter as unknown as { requestAdapterInfo?: () => Promise<GPUAdapterInfo> })
    .requestAdapterInfo;
  if (legacy) return legacy.call(adapter);
  return {} as GPUAdapterInfo;
}

function formatAdapterLabel(info: GPUAdapterInfo): string {
  const parts = [info.description, info.vendor, info.architecture].filter(Boolean);
  return parts.length ? parts.join(' · ') : 'Unknown GPU';
}
