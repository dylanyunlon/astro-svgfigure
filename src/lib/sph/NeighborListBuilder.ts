// === src/lib/sph/NeighborListBuilder.ts ===

import { SpatialHashGrid } from './SpatialHashGrid';
import { MAX_PARTICLES, MAX_NEIGHBORS, NeighborCSR } from './types';

export class NeighborListBuilder {
  private device: GPUDevice;
  private grid: SpatialHashGrid;

  private offsetCPU: Int32Array;
  private listCPU: Int32Array;

  offsetBuf: GPUBuffer;
  listBuf: GPUBuffer;

  constructor(device: GPUDevice) {
    this.device = device;
    this.grid = new SpatialHashGrid();

    this.offsetCPU = new Int32Array(MAX_PARTICLES + 1);
    this.listCPU   = new Int32Array(MAX_PARTICLES * MAX_NEIGHBORS);

    this.offsetBuf = device.createBuffer({
      label: 'nlb-offset',
      size: (MAX_PARTICLES + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.listBuf = device.createBuffer({
      label: 'nlb-list',
      size: MAX_PARTICLES * MAX_NEIGHBORS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
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

    // upload offset array (n+1 entries)
    this.device.queue.writeBuffer(
      this.offsetBuf, 0,
      this.offsetCPU.buffer, 0, (n + 1) * 4,
    );

    // upload list array (total neighbour count entries, min 4 bytes)
    const listBytes = Math.max(cursor * 4, 4);
    this.device.queue.writeBuffer(
      this.listBuf, 0,
      this.listCPU.buffer, 0, listBytes,
    );
  }

  getCSR(): NeighborCSR {
    return {
      offsetBuf: this.offsetBuf,
      listBuf:   this.listBuf,
      offset:    this.offsetCPU,
      list:      this.listCPU,
    };
  }

  destroy(): void {
    this.offsetBuf.destroy();
    this.listBuf.destroy();
  }
}
