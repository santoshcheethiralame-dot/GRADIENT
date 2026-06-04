/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />

// Allow importing WGSL shader source as a raw string: `import src from './x.wgsl?raw'`
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}
