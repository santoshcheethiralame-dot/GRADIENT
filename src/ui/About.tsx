import { Fragment } from 'react';

const REPO = 'https://github.com/santoshcheethiralame-dot/GRADIENT';
const PIPELINE = ['GPUDevice', 'GpuTensor', 'tiled matmul', 'forward', 'backprop', 'Adam', 'MNIST'];

const STATS = [
  { v: '49/49', l: 'self-test checks pass' },
  { v: '~97%', l: 'MNIST test accuracy' },
  { v: '5–20×', l: 'GPU vs. 1 CPU thread' },
  { v: '0', l: 'servers · 100% in-browser' },
];

const MLP_PHASES = [
  'device · GpuTensor · tiled matmul',
  'forward ops — relu, softmax, cross-entropy',
  'gradient-checked backward pass',
  'optimizers — SGD, then Adam',
  'MNIST → packed GPU buffers',
  'Worker-driven live dashboard',
];
const GPT_STEPS = [
  'architecture + autoregressive generation',
  'gradient-checked backprop + training',
  'full forward + backward in WGSL (~1e-7)',
  'GPU Adam — trains Shakespeare or your own text',
  'CPU vs GPU race — one CPU thread vs your GPU',
];

const WGSL = `// causal_softmax.wgsl — the attention core
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)
        gid: vec3<u32>) {
  let i   = gid.x;     // query row
  let lim = i + 1u;    // keys 0..i only
  var m = -1e30;
  for (var j = 0u; j < lim; j++) {
    m = max(m, inp[i*T + j] * scale);
  }
  // softmax over the causal window;
  // future keys (j > i) are masked to 0
}`;

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
    body: 'A from-scratch char transformer — causal self-attention + MLP, hand-derived backprop — trained end to end on the GPU. Paste your own text and watch it learn to write while the attention matrix sharpens.',
  },
  {
    tag: 'race',
    title: 'CPU vs GPU, live',
    body: 'The same transformer, same weights, trained two ways at once — one CPU thread vs your GPU in WGSL. The GPU laps it, taking many more steps and diving to far lower loss while the CPU crawls.',
  },
  {
    tag: 'backprop',
    title: 'See gradients flow',
    body: 'Most demos show attention — the forward pass. This shows the backward one: every layer’s gradient magnitude, live, ordered from the loss at the LM head back to the embeddings, computed by the same GPU backward that trains the net.',
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
        <div className="about-hero-main">
          <span className="about-kicker">▌ about · gradient</span>
          <h1 className="about-title">
            a neural network that trains on your <span className="hl">gpu</span>, in a browser tab.
          </h1>
          <p className="about-lede">
            No Python, no server, no WASM. Every FLOP — the forward pass, backpropagation, and the
            optimizer — is a WebGPU compute shader written in WGSL and dispatched from TypeScript.
            Open the page and watch the network learn, live, on your own graphics card.
          </p>
          <div className="about-cta">
            <button className="btn" onClick={onEnter}>
              open the lab →
            </button>
            <a className="btn ghost" href={REPO} target="_blank" rel="noreferrer">
              source on github
            </a>
          </div>
        </div>
        <aside className="codepanel">
          <div className="codepanel-bar">
            <i />
            <i />
            <i />
            <span>causal_softmax.wgsl</span>
          </div>
          <pre className="codepanel-code">{WGSL}</pre>
        </aside>
      </section>

      <div className="about-stats">
        {STATS.map((s) => (
          <div className="about-stat" key={s.l}>
            <b>{s.v}</b>
            <span>{s.l}</span>
          </div>
        ))}
      </div>

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
        <div className="proof-row">
          <p className="about-body">
            Numerical code is only as good as its tests. On every page load, gradient runs a
            self-test: each GPU kernel is computed and diffed against an independent <b>f64 CPU
            oracle</b> across a spread of shapes, and the entire backward pass is gated by{' '}
            <b>numerical gradient checking</b> (analytic vs. central-difference). Nothing is asserted
            — it's measured, in your browser.
          </p>
          <div className="proof-big">
            <b>49 / 49</b>
            <span>kernel + gradient checks · rel. err &lt; 1e-3</span>
          </div>
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
          Shipped as small, individually-verified steps — each landed only after its self-test
          passed. The WebGPU MLP engine came first, then a transformer was layered on top and trained
          end to end on the GPU.
        </p>
        <div className="build-cols">
          <div className="build-col">
            <h4>WebGPU MLP · 6 phases</h4>
            <ol>
              {MLP_PHASES.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ol>
          </div>
          <div className="build-col">
            <h4>nano-GPT · trained on the GPU</h4>
            <ol>
              {GPT_STEPS.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ol>
          </div>
        </div>
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
