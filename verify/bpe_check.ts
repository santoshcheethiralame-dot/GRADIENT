import { BpeTokenizer } from '../src/nn/bpe';

const corpus =
  'to be, or not to be, that is the question. to be is to do. to do is to be. ' +
  'the slings and arrows of outrageous fortune. to be or not to be. ';
const tok = new BpeTokenizer(corpus, 70);
const ids = tok.encode(corpus);
const round = tok.decode(ids);

console.log('baseChars', tok.chars.length, 'merges', tok.merges, 'vocab', tok.vocab);
console.log('roundtrip identity:', round === corpus ? 'PASS' : 'FAIL');
console.log('compression:', corpus.length, 'chars ->', ids.length, 'tokens', `(${(corpus.length / ids.length).toFixed(2)}x)`);
const sample = 'to be or not';
console.log(
  `"${sample}" ->`,
  tok.encode(sample).map((i) => JSON.stringify(tok.decode([i]))).join(' '),
);
