// The About page — a portfolio-facing overview of what gradient is, how it
// works, and how it's verified. Pure presentation; no GPU work here.

import { Fragment } from 'react';

const REPO = 'https://github.com/santoshcheethiralame-dot/GRADIENT';
const PIPELINE = ['GPUDevice', 'GpuTensor', 'tiled matmul', 'forward', 'backprop', 'Adam', 'MNIST'];

interface Feature {
  tag: string;
  title: string;
  body: string;
}

const FEATURES: Feature[] = [
  {
    tag: 'mlp',
    title: 'MNIST, end to end',
    body: 'A 2-layer net trains on MNIST in a Web Worker — tiled matmul, ReLU, softmax + cross-entropy, Adam — reaching ~97% test accuracy without ever touching the main thread.',
  },
  {
    tag: 'transformer',
    title: 'nano-GPT that writes',
    body: 'A from-scratch char transformer: causal self-attention + MLP, hand-derived backprop verified by gradient checking. Train it live and watch it reproduce a sentence.',
  },
  {
    tag: 'visualize',
    title: 'See it think',
    body: 'Live loss/accuracy traces, activation heatmaps, a PCA embedding projector that pulls the digit classes apart, and a sampled loss-landscape surface around the weights.',
  },
  {
    tag: 'interact',
    title: 'Draw a digit',
    body: 'Sketch a number and the trained net classifies it live on the GPU — full MNIST preprocessing (bounding-box crop, center-of-mass centering) runs in the browser.',
  },
  {
    tag: 'profile',
    title: 'GPU profiler',
    body: 'Amortized kernel timing across matmul sizes (naive vs. shared-memory tiled) and a full train step — measured over many back-to-back dispatches, no readback stalls.',
  },
  {
    tag: 'fallback',
    title: 'Runs anywhere',
    body: 'No WebGPU? An f64 CPU engine — the same one used as the correctness oracle — trains a smaller net so the demo works in any browser. Append ?cpu=1 to force it.',
  },
];

const STACK = ['WebGPU / WGSL', 'TypeScript', 'React 19', 'Vite', 'Web Workers', 'Canvas 2D'];

export function About({ onEnter }: { onEnter: () => void }) {
  return (
    <div className="about">
      <section className="card about-hero">
        <span className="about-kicker">▌ about · gradient</span>
        <h1 className="about-title">
          A neural network that trains on your <span className="hl">GPU</span>, in a browser tab.
        </h1>
        <p className="about-lede">
          No Python, no server, no WASM. Every FLOP — the forward pass, backpropagation, and the
          optimizer — is a WebGPU compute shader written in WGSL and dispatched from TypeScript.
          Open the page and watch the network learn, live, on your own graphics card.
        </p>
        <div className="about-cta">
          <button className="btn primary" onClick={onEnter}>
            open the lab →
          </button>
          <a className="btn ghost" href={REPO} target="_blank" rel="noreferrer">
            source on github
          </a>
        </div>
      </section>

      <section className="card about-sec">
        <h2>
          <span className="ch">01</span>built from the metal up
        </h2>
        <p className="about-body">
          Modern training runs on the GPU, but the GPU is usually a black box behind a framework.
          gradient rebuilds the stack by hand — explicit buffers, bind groups, compute pipelines, and
          16×16 shared-memory tiling — so every step is visible and inspectable. One device, one
          queue; tensors are allocated once and reused across steps.
        </p>
        <div className="pipeline">
          {PIPELINE.map((s, i) => (
            <Fragment key={s}>
              <span className="pstep">{s}</span>
              {i < PIPELINE.length - 1 && <span className="parrow">→</span>}
            </Fragment>
          ))}
        </div>
      </section>

      <section className="card about-sec about-proof">
        <h2>
          <span className="ch">02</span>proven correct, every load
        </h2>
        <p className="about-body">
          Numerical code is only as good as its tests. On every page load, gradient runs a self-test:
          each GPU kernel is computed and diffed against an independent <b>f64 CPU oracle</b> across a
          spread of shapes, and the entire backward pass is gated by <b>numerical gradient checking</b>{' '}
          (analytic vs. central-difference). Nothing is asserted — it's measured, in your browser.
        </p>
        <div className="proof-stat">
          <b>44 / 44</b>
          <span>kernel + gradient checks pass · rel. err &lt; 1e-3</span>
        </div>
      </section>

      <section className="about-sec-bare">
        <h2 className="about-h2">
          <span className="ch">03</span>what's inside
        </h2>
        <div className="feat-grid">
          {FEATURES.map((f) => (
            <div className="feat card" key={f.tag}>
              <span className="feat-tag">{f.tag}</span>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card about-sec">
        <h2>
          <span className="ch">04</span>built in the open
        </h2>
        <p className="about-body">
          Shipped as small, individually-verified steps — six phases for the WebGPU MLP engine
          (device → tensors → matmul → forward → gradient-checked backprop → optimizers → MNIST →
          dashboard), then four increments for the transformer (architecture → backprop → live panel
          → WGSL forward port). Each landed only after its self-test passed.
        </p>
        <div className="about-stack">
          {STACK.map((s) => (
            <span key={s}>{s}</span>
          ))}
        </div>
      </section>

      <footer className="about-foot">
        <span>gradient — a GPU neural-net trainer in the browser.</span>
        <a href={REPO} target="_blank" rel="noreferrer">
          github.com/santoshcheethiralame-dot/GRADIENT
        </a>
      </footer>
    </div>
  );
}
