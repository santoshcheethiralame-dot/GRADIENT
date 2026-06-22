# verify — independent gradient check against PyTorch

gradient's transformer backward is hand-derived (no autodiff) and runs in WGSL on
the GPU. In-app it's gradient-checked against finite differences and diffed against
an f64 CPU oracle on every load. This folder adds a third, independent witness:
**PyTorch autograd**.

- `export_reference.ts` builds a fixed-seed nano-GPT (`src/nn/nanogpt.ts`), runs one
  forward + backward, and dumps the weights, input, loss and all 17 parameter
  gradients to `reference.json`.
- `verify_pytorch.py` rebuilds the **identical** architecture in PyTorch (same pre-LN
  block, single-head causal attention, tanh-GELU MLP, biased layer-norm, mean
  cross-entropy), loads those exact weights, runs autograd, and diffs its gradients
  against the hand-derived ones.

## Run

```bash
npx tsx verify/export_reference.ts     # writes verify/reference.json
python verify/verify_pytorch.py        # needs: pip install torch
```

## Result

```
loss   js=4.5484059113   torch=4.5484059634   |diff|=5.21e-08
max abs err:              4.15e-07
max rel err (floor 1e-3): 5.58e-05
PASS — hand-derived gradients match PyTorch autograd to f32 precision
```

Every gradient agrees with PyTorch to f32 precision (~1e-7 absolute; the loss to
~5e-8). The relative-error floor of 1e-3 avoids the usual artifact where a tiny
absolute difference on a near-zero gradient reports a huge ratio.

This is verification only — the app itself uses no Python and no PyTorch.
