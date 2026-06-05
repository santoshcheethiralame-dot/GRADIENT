// Message contract between the main thread (UI) and the training worker.
// WebGPU lives in the worker, so all data crossing back is plain CPU arrays /
// scalars. Float32Arrays are sent as transferables to avoid copies.

export type InMsg =
  | { type: 'init' }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset'; hidden: number; lr: number }
  | { type: 'infer'; pixels: Float32Array } // [pixels] normalized, drawn digit
  | { type: 'landscape' }; // compute a loss surface around the current weights

export interface ReadyMsg {
  type: 'ready';
  adapter: string;
  train: number;
  test: number;
  pixels: number;
  hidden: number;
  lr: number;
}
export interface StatusMsg {
  type: 'status';
  message: string;
}
export interface MetricsMsg {
  type: 'metrics';
  step: number;
  loss: number;
  trainAcc: number;
  stepsPerSec: number;
}
export interface TestAccMsg {
  type: 'testacc';
  step: number;
  testAcc: number;
}
export interface EmbeddingMsg {
  type: 'embedding';
  coords: Float32Array; // [n*2] hidden activations projected to 2-D (PCA), in ~[-1,1]
  labels: Uint8Array; // [n] true class per point
  step: number;
}
export interface ActivationsMsg {
  type: 'activations';
  input: Float32Array; // [pixels] the probe digit (normalized)
  hidden: Float32Array; // [hidden] ReLU activations
  probs: Float32Array; // [10] softmax
  label: number;
  pred: number;
}
export interface ProbsMsg {
  type: 'probs';
  probs: Float32Array; // [10] response to an infer request
}
export interface LandscapeMsg {
  type: 'landscape';
  grid: Float32Array; // [size*size] loss sampled on a 2-D grid around the weights
  size: number;
  step: number;
}
export interface ErrorMsg {
  type: 'error';
  message: string;
}

export type OutMsg =
  | ReadyMsg
  | StatusMsg
  | MetricsMsg
  | TestAccMsg
  | EmbeddingMsg
  | ActivationsMsg
  | ProbsMsg
  | LandscapeMsg
  | ErrorMsg;
