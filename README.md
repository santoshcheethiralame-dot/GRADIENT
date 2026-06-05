# gradient

**A neural network that trains entirely on your GPU, in the browser** — no Python, no server, no WASM. The forward pass, backpropagation, and the optimizer are all WebGPU compute shaders (WGSL) dispatched from TypeScript. Open the page and watch it learn, live, on your own graphics card.

**[Live demo](https://santoshcheethiralame-dot.github.io/GRADIENT/)** · WebGPU / WGSL · React + TypeScript

> **Status: complete.** The full stack runs on WebGPU and trains two models — a 2-layer MLP on **MNIST** (~97% test accuracy) and a from-scratch **nano-GPT** char transformer that learns to write. Every GPU kernel is diffed against an f64 CPU oracle and the backward pass is numerical-gradient-checked, all re-run on each page load: **44/44 checks pass**.

---

## What it is

Modern training runs on the GPU, but the GPU is usually a black box behind a framework. `gradient` rebuilds the stack from the metal up — explicit buffers, bind groups, compute pipelines, and 16×16 shared-memory tiling — so every FLOP is visible and inspectable in a browser tab. It trains:

- a **2-layer MLP on MNIST** (~97% test accuracy) in a **Web Worker**, streaming live metrics and activations to the dashboard; and
- a from-scratch **nano-GPT** — a char-level transformer (causal self-attention + MLP) with hand-derived, **gradient-checked** backprop that trains to reproduce a sentence.

## Proven correct, every load

Numerical code is only as good as its tests. On every page load `gradient` runs a self-test: each GPU kernel is computed and diffed against an independent **f64 CPU oracle** across a spread of shapes (including non-tile-multiples, to exercise boundary masking), and the entire backward pass is gated by **numerical gradient checking** (analytic vs. central-difference). **44/44 checks pass** at rel. err < 1e-3 — measured in your browser, not asserted.

## Features

- **Live training dashboard** — loss/accuracy traces and hidden-activation heatmaps at 60fps
- **nano-GPT** — train a char transformer live and watch its greedy output sharpen into the target sentence as the loss falls
- **Embedding projector** — hidden activations → PCA → 2-D scatter; watch the digit classes pull apart as the net learns
- **Loss landscape** — the loss surface sampled on a grid around the weights along two filter-normalized random directions
- **GPU profiler** — amortized kernel throughput (naive vs. tiled matmul across sizes, plus a full train step) measured over many back-to-back dispatches
- **Draw-a-digit** — sketch a number and the trained net classifies it live, with full MNIST preprocessing (bounding-box crop → 20px scale → center-of-mass centering)
- **CPU fallback** — no WebGPU? The same f64 engine used as the correctness oracle trains a smaller net so the demo runs anywhere (append `?cpu=1` to force it)
- **About page** — an overview of how it works and how it's verified

## Stack

- **WebGPU / WGSL** — every math kernel: matmul (naive + tiled), bias-add, ReLU/sigmoid/GELU, softmax, cross-entropy, layer-norm, causal-masked attention softmax, the backward kernels, and SGD/Adam
- **TypeScript** — buffer/pipeline orchestration and the `GpuTensor` memory model
- **React 19 + Vite** — UI shell; self-hosted Archivo + JetBrains Mono
- **Web Worker** — the training loop runs off the main thread, streaming to the UI
- **Canvas 2D** — loss/accuracy charts, activation heatmaps, the embedding scatter, and the pseudo-3D loss surface

## Quick start

**Prerequisites:** [Node.js](https://nodejs.org) 18 or newer, and a WebGPU-capable browser (Chrome / Edge 113+, Safari 18+, or Firefox 141+). No compatible GPU? The app falls back to a CPU engine automatically, so it still runs anywhere.

```bash
git clone https://github.com/santoshcheethiralame-dot/GRADIENT.git
cd GRADIENT
npm install
npm run dev          # → http://localhost:5173
```

Open the URL — the app boots, runs its kernel self-test, and starts training. The other scripts:

```bash
npm run build        # type-check + production bundle → dist/
npm run preview      # serve the production build locally
```

(If you use [nvm](https://github.com/nvm-sh/nvm), `nvm use` picks up the pinned Node version from `.nvmrc`.)

## Deploy

**GitHub Pages — included.** The workflow at `.github/workflows/deploy.yml` builds and publishes on every push to `main`. To enable it on your fork: **Settings → Pages → Build and deployment → Source: _GitHub Actions_**, then push (or run the workflow manually). It publishes to `https://<your-username>.github.io/<repo>/`; the workflow sets Vite's base path from the repo name, so forks of any name work without edits.

**Vercel or Netlify — zero config.** Both auto-detect Vite, so just import the repository and deploy — no build settings to fill in (these serve from the domain root).

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fsantoshcheethiralame-dot%2FGRADIENT)
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/santoshcheethiralame-dot/GRADIENT)

## Troubleshooting

- **"WebGPU not available."** Use a supported browser, or just run it as-is — the app drops to the CPU engine. Force CPU mode any time with `?cpu=1` in the URL.
- **Blank page or 404 assets on GitHub Pages.** Set the Pages source to **GitHub Actions** (not a branch) — only that path runs the build that sets the correct base path.
- **Port 5173 already in use.** `npm run dev -- --port 5174`.
- **Node version errors.** This needs Node 18+ (`.nvmrc` pins 20).

## How it was built

Shipped as small, individually-verified steps — each landed only after its self-test passed.

**WebGPU MLP engine — 6 phases**

1. **Foundation** — device, `GpuTensor`, naive + tiled matmul, self-test
2. **Forward** — bias-add, ReLU, sigmoid, softmax + cross-entropy
3. **Backward** — gradient shaders + numerical gradient checking
4. **Optimizers** — SGD, then Adam; training loop on synthetic data
5. **Data** — MNIST IDX parsing → packed GPU buffers, batched gather
6. **Visualize** — Worker-driven training, live dashboard, draw-a-digit

**nano-GPT — 4 increments**

1. **Architecture** — tokenizer, embeddings, a pre-LN block (attention + MLP), generation
2. **Backprop + training** — hand-derived gradients through softmax-attention and layer-norm, gradient-checked, then trained to overfit and reproduce a sentence
3. **GPU forward port** — layer-norm / causal-softmax / GELU / residual kernels in WGSL (projections reuse the matmul kernels), verified GPU-vs-CPU to ~1e-7
4. **Live panel** — train it in the browser and watch it write

## Architecture notes

- One `GPUDevice`, one `GPUQueue`. Submissions are ordered, so forward / backward / update share the queue without explicit synchronization.
- Tensors are allocated once and reused across steps — the only per-step CPU touch is reading back the scalar loss for logging.
- Shaders live as `.wgsl` files imported with Vite's `?raw` suffix, so the real kernel source is first-class and inspectable.
- `reference.ts` (the f64 CPU implementation) pulls double duty: the correctness oracle for the self-test **and** the no-WebGPU fallback engine.

## License

MIT — see [LICENSE](LICENSE).
