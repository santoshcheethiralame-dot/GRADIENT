import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// WGSL shaders are imported with the `?raw` suffix (built into Vite), so no
// custom plugin is needed. The worker is bundled via the native `?worker` /
// `new Worker(new URL(...), import.meta.url)` syntax Vite understands.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    host: true,
  },
  worker: {
    format: 'es',
  },
});
