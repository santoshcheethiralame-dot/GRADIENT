import type { NanoGptConfig } from '../nn/nanogpt';

export type RaceStart = {
  type: 'start';
  cfg: NanoGptConfig;
  seed: number;
  data: number[];
  T: number;
  lr: number;
  durationMs: number;
  reportMs: number;
};

export type RaceStop = { type: 'stop' };

export type RaceIn = RaceStart | RaceStop;

export type RaceProgress = { type: 'progress'; step: number; loss: number; elapsedMs: number };

export type RaceDone = { type: 'done'; step: number; loss: number; elapsedMs: number };

export type RaceOut = RaceProgress | RaceDone;
