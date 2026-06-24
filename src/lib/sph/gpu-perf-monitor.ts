/**
 * gpu-perf-monitor.ts — FPS counter + per-pass timing + drawcall counting
 */









export class GPUPerfMonitor {
  private frameTimes: number[] = [];
  private maxSamples = 60;
  private passTimings: Map<string, number> = new Map();
  private drawCallCount = 0;
  private _lastFrame = 0;

  /** Call at start of frame */
  frameStart(): void {
    this._lastFrame = performance.now();
    this.drawCallCount = 0;
    this.passTimings.clear();
  }

  /** Call at end of frame */
  frameEnd(): void {
    const dt = performance.now() - this._lastFrame;
    this.frameTimes.push(dt);
    if (this.frameTimes.length > this.maxSamples) this.frameTimes.shift();
  }

  /** Wrap around a pass to measure its CPU time */
  passStart(name: string): number {
    return performance.now();
  }

  passEnd(name: string, startTime: number): void {
    this.passTimings.set(name, performance.now() - startTime);
  }

  /** Call this to count a draw call */
  countDraw(): void {
    this.drawCallCount++;
  }

  get fps(): number {
    if (this.frameTimes.length === 0) return 0;
    const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    return avg > 0 ? 1000 / avg : 0;
  }

  get avgFrameMs(): number {
    if (this.frameTimes.length === 0) return 0;
    return this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
  }

  get stats(): {
    fps: number; frameMs: number; drawCalls: number;
    passes: Record<string, number>;
  } {
    return {
      fps: Math.round(this.fps),
      frameMs: Math.round(this.avgFrameMs * 100) / 100,
      drawCalls: this.drawCallCount,
      passes: Object.fromEntries(this.passTimings),
    };
  }
}
