import { AABB } from './aabb-manager';

interface Endpoint {
  value: number;
  bodyId: number;
  isMin: boolean;
}

interface BodyEntry {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export class SortAndSweep {
  private axisX: Endpoint[] = [];
  private axisY: Endpoint[] = [];
  private bodyMap: Map<number, BodyEntry> = new Map();

  private insertSorted(axis: Endpoint[], ep: Endpoint): void {
    axis.push(ep);
    let i = axis.length - 1;
    while (i > 0 && axis[i - 1].value > axis[i].value) {
      const tmp = axis[i - 1];
      axis[i - 1] = axis[i];
      axis[i] = tmp;
      i--;
    }
  }

  private insertionSort(axis: Endpoint[]): void {
    for (let i = 1; i < axis.length; i++) {
      const key = axis[i];
      let j = i - 1;
      while (j >= 0 && axis[j].value > key.value) {
        axis[j + 1] = axis[j];
        j--;
      }
      axis[j + 1] = key;
    }
  }

  private removeFromAxis(axis: Endpoint[], bodyId: number): void {
    for (let i = axis.length - 1; i >= 0; i--) {
      if (axis[i].bodyId === bodyId) {
        axis.splice(i, 1);
      }
    }
  }

  updateBody(bodyId: number, aabb: AABB): void {
    const existing = this.bodyMap.get(bodyId);

    if (existing) {
      // Update values in place for temporal coherence (avoid full re-insert)
      for (const ep of this.axisX) {
        if (ep.bodyId === bodyId) {
          ep.value = ep.isMin ? aabb.minX : aabb.maxX;
        }
      }
      for (const ep of this.axisY) {
        if (ep.bodyId === bodyId) {
          ep.value = ep.isMin ? aabb.minY : aabb.maxY;
        }
      }
      // Insertion sort preserves nearly-sorted order efficiently
      this.insertionSort(this.axisX);
      this.insertionSort(this.axisY);
    } else {
      // New body: insert endpoints into sorted arrays
      this.insertSorted(this.axisX, { value: aabb.minX, bodyId, isMin: true });
      this.insertSorted(this.axisX, { value: aabb.maxX, bodyId, isMin: false });
      this.insertSorted(this.axisY, { value: aabb.minY, bodyId, isMin: true });
      this.insertSorted(this.axisY, { value: aabb.maxY, bodyId, isMin: false });
    }

    this.bodyMap.set(bodyId, {
      xMin: aabb.minX,
      xMax: aabb.maxX,
      yMin: aabb.minY,
      yMax: aabb.maxY,
    });
  }

  removeBody(bodyId: number): void {
    if (!this.bodyMap.has(bodyId)) return;
    this.removeFromAxis(this.axisX, bodyId);
    this.removeFromAxis(this.axisY, bodyId);
    this.bodyMap.delete(bodyId);
  }

  private sweepAxis(axis: Endpoint[]): Set<string> {
    const overlapping = new Set<string>();
    const active = new Set<number>();

    for (const ep of axis) {
      if (ep.isMin) {
        for (const otherId of active) {
          const key =
            ep.bodyId < otherId
              ? `${ep.bodyId}:${otherId}`
              : `${otherId}:${ep.bodyId}`;
          overlapping.add(key);
        }
        active.add(ep.bodyId);
      } else {
        active.delete(ep.bodyId);
      }
    }

    return overlapping;
  }

  computePairs(): [number, number][] {
    const xPairs = this.sweepAxis(this.axisX);
    const yPairs = this.sweepAxis(this.axisY);

    const result: [number, number][] = [];

    for (const key of xPairs) {
      if (yPairs.has(key)) {
        const [a, b] = key.split(':').map(Number);
        result.push([a, b]);
      }
    }

    return result;
  }
}
