// === src/lib/sph/NeighborListBuilder.ts ===
// [M1150] WebGL2 compat: all WebGPU concrete types replaced with `any`
// NeighborCSR inlined to avoid cross-module interface resolution issues with esbuild

import { SpatialHashGrid } from './SpatialHashGrid';
import { MAX_PARTICLES, MAX_NEIGHBORS } from './types';

// Inline NeighborCSR to avoid types.ts module-resolution issues in esbuild/Vite
export interface NeighborCSR {
  offsetBuf: any; // GPUBuffer | WebGLBuffer | null
  listBuf:   any; // GPUBuffer | WebGLBuffer | null
  offsetCPU: Int32Array;
  listCPU:   Int32Array;
}

export class NeighborListBuilder {
  private device: any; // GPUDevice | null
  private grid: SpatialHashGrid;

  private offsetCPU: Int32Array;
  private listCPU:   Int32Array;

  offsetBuf: any; // GPUBuffer | null
  listBuf:   any; // GPUBuffer | null

  constructor(device: any) {
    this.device = device;
    this.grid = new SpatialHashGrid();

    this.offsetCPU = new Int32Array(MAX_PARTICLES + 1);
    this.listCPU   = new Int32Array(MAX_PARTICLES * MAX_NEIGHBORS);

    // Only allocate GPU buffers when a real WebGPU device is present
    if (device && typeof device.createBuffer === 'function') {
      const GPU_STORAGE  = 0x0088; // GPUBufferUsage.STORAGE
      const GPU_COPY_DST = 0x0008; // GPUBufferUsage.COPY_DST
      this.offsetBuf = device.createBuffer({
        label: 'nlb-offset',
        size: (MAX_PARTICLES + 1) * 4,
        usage: GPU_STORAGE | GPU_COPY_DST,
      });
      this.listBuf = device.createBuffer({
        label: 'nlb-list',
        size: MAX_PARTICLES * MAX_NEIGHBORS * 4,
        usage: GPU_STORAGE | GPU_COPY_DST,
      });
    } else {
      this.offsetBuf = null;
      this.listBuf   = null;
    }
  }

  build(px: Float32Array, py: Float32Array, n: number, radius: number): void {
    this.grid.build(px, py, n);

    const tmp = new Int32Array(MAX_NEIGHBORS);
    let cursor = 0;

    this.offsetCPU[0] = 0;

    for (let i = 0; i < n; i++) {
      const cnt    = this.grid.queryRadius(px[i], py[i], radius, px, py, n, tmp);
      const actual = Math.min(cnt, MAX_NEIGHBORS);

      for (let k = 0; k < actual; k++) {
        this.listCPU[cursor + k] = tmp[k];
      }

      cursor += actual;
      this.offsetCPU[i + 1] = cursor;
    }

    // Upload to GPU only when buffers exist
    if (this.offsetBuf && this.device?.queue) {
      this.device.queue.writeBuffer(
        this.offsetBuf, 0,
        this.offsetCPU.buffer, 0, (n + 1) * 4,
      );

      const listBytes = Math.max(cursor * 4, 4);
      this.device.queue.writeBuffer(
        this.listBuf, 0,
        this.listCPU.buffer, 0, listBytes,
      );
    }
  }

  getCSR(): NeighborCSR {
    return {
      offsetBuf: this.offsetBuf,
      listBuf:   this.listBuf,
      offsetCPU: this.offsetCPU,
      listCPU:   this.listCPU,
    };
  }

  destroy(): void {
    if (this.offsetBuf && typeof this.offsetBuf.destroy === 'function') {
      this.offsetBuf.destroy();
    }
    if (this.listBuf && typeof this.listBuf.destroy === 'function') {
      this.listBuf.destroy();
    }
  }
}
