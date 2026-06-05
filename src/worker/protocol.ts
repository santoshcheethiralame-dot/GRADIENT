export type InMsg =
  | { type: 'init' }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset'; hidden: number; lr: number }
  | { type: 'infer'; pixels: Float32Array }
  | { type: 'landscape' };

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
  coords: Float32Array;
  labels: Uint8Array;
  step: number;
}
export interface ActivationsMsg {
  type: 'activations';
  input: Float32Array;
  hidden: Float32Array;
  probs: Float32Array;
  label: number;
  pred: number;
}
export interface ProbsMsg {
  type: 'probs';
  probs: Float32Array;
}
export interface LandscapeMsg {
  type: 'landscape';
  grid: Float32Array;
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
