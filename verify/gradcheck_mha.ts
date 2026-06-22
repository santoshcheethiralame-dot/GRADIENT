import { NanoGpt, CharTokenizer } from '../src/nn/nanogpt';

const tok = new CharTokenizer('the quick brown fox jumps over the lazy dog. ');
const full = tok.encode('the quick br');

for (const nHeads of [1, 2, 4]) {
  const cfg = { vocab: tok.vocab, dEmbed: 16, dFF: 32, blockSize: 8, nHeads };
  const model = new NanoGpt(cfg, 2, 0.5);
  const T = cfg.blockSize;
  const ids = full.slice(0, T);
  const targets = full.slice(1, T + 1);
  model.zeroGrad();
  model.backward(model.forwardCache(ids), targets);
  const eps = 1e-3;
  const GT = 1e-2;
  let worst = 0;
  let checked = 0;
  for (const p of model.params()) {
    for (let e = 0; e < p.w.length; e++) {
      if (Math.abs(p.g[e]) < GT) continue;
      const o = p.w[e];
      p.w[e] = o + eps;
      const lp = model.forwardLoss(ids, targets);
      p.w[e] = o - eps;
      const lm = model.forwardLoss(ids, targets);
      p.w[e] = o;
      const num = (lp - lm) / (2 * eps);
      const rel = Math.abs(num - p.g[e]) / (Math.abs(num) + Math.abs(p.g[e]) + 1e-12);
      worst = Math.max(worst, rel);
      checked++;
    }
  }
  console.log(
    `nHeads=${nHeads}  checked=${checked}  worst rel=${worst.toExponential(2)}  -> ${worst < 2e-2 ? 'PASS' : 'FAIL'}`,
  );
}
