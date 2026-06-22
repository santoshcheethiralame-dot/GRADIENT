import { writeFileSync } from 'node:fs';
import { NanoGpt, CharTokenizer } from '../src/nn/nanogpt';

const corpus = 'the quick brown fox jumps over the lazy dog. ';
const tok = new CharTokenizer(corpus);
const cfg = { vocab: tok.vocab, dEmbed: 16, dFF: 32, blockSize: 8 };

const model = new NanoGpt(cfg, 4, 0.4);
const T = cfg.blockSize;
const full = tok.encode('the quick br');
const ids = full.slice(0, T);
const targets = full.slice(1, T + 1);

const loss = model.forwardLoss(ids, targets);
model.zeroGrad();
model.backward(model.forwardCache(ids), targets);

const params = model.params();
const out = {
  cfg,
  ids,
  targets,
  loss,
  weights: Object.fromEntries(params.map((p) => [p.name, Array.from(p.w)])),
  grads: Object.fromEntries(params.map((p) => [p.name, Array.from(p.g)])),
};

writeFileSync(new URL('./reference.json', import.meta.url), JSON.stringify(out));
console.log('wrote verify/reference.json', { vocab: cfg.vocab, T, params: params.length, loss });
