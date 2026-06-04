# gradient

A neural network that trains **entirely on your GPU, in the browser** — no Python, no server, no WASM. Forward pass, backpropagation, and the optimizer are all WebGPU compute shaders (WGSL) dispatched from TypeScript. The UI visualizes activations and gradients live as the network learns.

> Status: **Complete — all 6 phases.** The full stack runs on WebGPU: device → tensors → matmul → forward → gradient-checked backprop → SGD/Adam → MNIST → a **Web Worker** that trains continuously off the main thread, streaming metrics and activations to a live **oscilloscope-style dashboard** (graticule loss/accuracy traces, blue→amber thermal activation heatmaps) plus a **draw-a-digit** demo that classifies your handwriting live (~**97% test accuracy**). 32/32 kernel checks pass against a CPU oracle, and backprop is verified by numerical gradient checking — all re-run on every page load. UI: a Tektronix/Grafana-inspired scientific-instrument theme.

## Why

Modern training runs on the GPU, but the GPU is usually a black box behind PyTorch. `gradient` rebuilds the stack from the metal up — explicit buffers, bind groups, compute pipelines, and shared-memory tiling — so every FLOP is visible and inspectable in a browser tab.

## Stack

- **WebGPU / WGSL** — compute pipelines for all math (matmul, activations, backprop, SGD/Adam)
- **TypeScript** — buffer/pipeline orchestration, the `GpuTensor` memory model
- **React + Vite** — UI shell, self-hosted Archivo + JetBrains Mono
- **Canvas 2D** — the oscilloscope loss/accuracy chart (graticule + phosphor glow) and the thermal activation heatmaps
- **Web Worker** — the training loop runs off the main thread, streaming to the UI

## Run

```bash
npm install
npm run dev      # http://localhost:5173
```

Requires a WebGPU-capable browser (Chrome / Edge 113+, recent Safari, or Firefox with the WebGPU flag). On load the app requests a GPU adapter and runs a kernel self-test: each matmul shape is computed on the GPU and compared against a CPU reference, with max absolute/relative error reported per kernel.

## Roadmap

1. **Foundation** — device, `GpuTensor`, naive + tiled matmul, self-test ✅
2. **Forward** — bias add, ReLU, sigmoid, softmax + cross-entropy ✅
3. **Backward** — gradient shaders + numerical gradient checking ✅
4. **Optimizers** — SGD, then Adam; training loop on synthetic data ✅
5. **Data** — MNIST IDX parsing → GPU buffers, batched gather ✅
6. **Visualize** — Worker-driven training, live instrument dashboard at 60fps, draw-a-digit ✅

## Architecture notes

- One `GPUDevice`, one `GPUQueue`. Submissions are ordered; forward/backward/update share the queue.
- Tensors are allocated once and reused across steps — the only per-step CPU touch is reading back the scalar loss for logging.
- Shaders live as `.wgsl` files imported with Vite's `?raw` suffix, so the real kernel source is first-class and inspectable.

## License

MIT
