const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [9, 13, 24]],
  [0.28, [28, 78, 150]],
  [0.52, [46, 157, 255]],
  [0.76, [255, 176, 64]],
  [0.92, [255, 214, 90]],
  [1.0, [255, 246, 214]],
];

export function ember(t: number): [number, number, number] {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1];
      const [b, cb] = STOPS[i];
      const u = (x - a) / (b - a || 1);
      return [
        Math.round(ca[0] + (cb[0] - ca[0]) * u),
        Math.round(ca[1] + (cb[1] - ca[1]) * u),
        Math.round(ca[2] + (cb[2] - ca[2]) * u),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export function emberCss(t: number): string {
  const [r, g, b] = ember(t);
  return `rgb(${r},${g},${b})`;
}
