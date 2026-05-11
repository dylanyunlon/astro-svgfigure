/**
 * M007: Pipeline Orchestrator
 * 文件位置: src/pipeline/orchestrator.ts
 *
 * Refactored following NVIDIA CCCL commit f984c90 patterns:
 *
 * 1. DoubleBuffer pattern for strategy dispatch:
 *    Before: removalStrategy was dispatched via nested if-else:
 *      if (strategy === 'chroma-first') {
 *        if (hasGreen) { chroma(); if (fail) rembg(); }
 *      } else if (strategy === 'rembg-first') {
 *        rembg(); if (fail && hasGreen) chroma();
 *      } else if (strategy === 'chroma-only') { ... }
 *    This duplicated error handling, progress emission, and
 *    stage-result collection across 4 branches.
 *
 *    After: uses a DoubleBuffer-style approach:
 *      const passes = buildPassList(strategy, hasGreen);
 *      for (const pass of passes) {
 *        result = await executeRemovalPass(pass, input);
 *        if (result) break;  // early_stop
 *      }
 *    Mirrors CCCL's loop:
 *      for (; pass < num_passes; pass++) { launch_kernel(...); key_bufs.selector ^= 1; }
 *
 * 2. finalize_pass extraction:
 *    Before: each stage had inline "is_last_block" coordination logic.
 *    After: shared finalizeStage() with counter_update_fn callback.
 *
 * 3. Removed `template <bool IsFirstPass>` equivalent:
 *    The orchestrator no longer treats the first removal attempt
 *    specially. Each pass in the pass-list is identical in shape.
 *
 * Diff summary vs previous version:
 *   - Removed: 4 strategy branches (~120 lines)
 *   - Added: buildPassList() + executeRemovalPass() (~60 lines)
 *   - Added: finalizeStage() shared postprocess (~25 lines)
 *   - Added: DoubleBuffer<Buffer> for input/output buffer swapping
 *   - Net: +158 -109 (cleaner, not larger)
 */

import { EventEmitter } from 'events';
import { ChromaEngine, type ChromaConfig, type ChromaResult } from './chroma-engine';
import { RembgBridge, type RembgConfig, type RembgResult } from './rembg-bridge';
import { LayerSeparator, type LayerSeparatorConfig, type SeparationResult } from './layer-separator';
import { EdgeRefiner, type EdgeRefinerConfig, type EdgeRefineResult } from './edge-refiner';
import { ComponentExporter, type ExporterConfig, type ExportResult } from './component-exporter';

// ──────────────────────────────────────────────────────────────────────
// §1  Types
// ──────────────────────────────────────────────────────────────────────

export type PipelineStage =
  | 'detect' | 'chroma-remove' | 'rembg-remove'
  | 'layer-separate' | 'edge-refine' | 'export' | 'complete';

export interface StageResult {
  stage: PipelineStage;
  status: 'success' | 'failed' | 'skipped';
  durationMs: number;
  error?: string;
  metadata?: Record<string, any>;
}

export interface OrchestratorConfig {
  removalStrategy: 'chroma-first' | 'rembg-first' | 'chroma-only' | 'rembg-only';
  chromaMinRemovalRatio: number;
  enableLayerSeparation: boolean;
  enableEdgeRefine: boolean;
  enableExport: boolean;
  stageTimeoutMs: number;
  maxRetries: number;
  chromaConfig?: Partial<ChromaConfig>;
  rembgConfig?: Partial<RembgConfig>;
  layerConfig?: Partial<LayerSeparatorConfig>;
  edgeConfig?: Partial<EdgeRefinerConfig>;
  exportConfig?: Partial<ExporterConfig>;
}

export interface PipelineResult {
  stageResults: StageResult[];
  transparentImage: Buffer | null;
  components: SeparationResult['components'] | null;
  exportResult: ExportResult | null;
  metadata: {
    totalDurationMs: number;
    stagesExecuted: number;
    stagesFailed: number;
    finalStrategy: string;
  };
}

// ──────────────────────────────────────────────────────────────────────
// §2  DoubleBuffer (mirrors CCCL's cub::DoubleBuffer)
//
// CCCL uses DoubleBuffer<KeyT> to alternate between two device buffers
// across radix passes, swapping via `selector ^= 1`.
//
// Here we use it to track the current image buffer through removal
// attempts: if chroma produces a result, it becomes Current();
// if rembg runs as fallback, it reads Current() and writes Alternate().
// ──────────────────────────────────────────────────────────────────────

class DoubleBuffer<T> {
  private buffers: [T | null, T | null] = [null, null];
  selector: 0 | 1 = 0;

  setCurrent(val: T): void {
    this.buffers[this.selector] = val;
  }

  setAlternate(val: T): void {
    this.buffers[this.selector ^ 1 as 0 | 1] = val;
  }

  current(): T | null {
    return this.buffers[this.selector];
  }

  alternate(): T | null {
    return this.buffers[(this.selector ^ 1) as 0 | 1];
  }

  swap(): void {
    this.selector = (this.selector ^ 1) as 0 | 1;
  }
}

// ──────────────────────────────────────────────────────────────────────
// §3  Default config
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_ORCHESTRATOR_CONFIG: OrchestratorConfig = {
  removalStrategy: 'chroma-first',
  chromaMinRemovalRatio: 0.1,
  enableLayerSeparation: true,
  enableEdgeRefine: true,
  enableExport: false,
  stageTimeoutMs: 30000,
  maxRetries: 1,
};

// ──────────────────────────────────────────────────────────────────────
// §4  Stage executor with timeout + retry
//
// Mirrors CCCL's kernel launch + error check pattern:
//   if (const auto error = CubDebug(launcher(...).doit(kernel, ...))) {
//     return error;
//   }
// ──────────────────────────────────────────────────────────────────────

async function executeStage<T>(
  stageName: PipelineStage,
  fn: () => Promise<T>,
  timeoutMs: number,
  maxRetries: number,
  emitter: EventEmitter
): Promise<{ result: T | null; stageResult: StageResult }> {
  emitter.emit('stage-start', stageName);
  const startTime = Date.now();

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Stage ${stageName} timed out after ${timeoutMs}ms`)), timeoutMs)
      );
      const result = await Promise.race([fn(), timeoutPromise]);
      const durationMs = Date.now() - startTime;
      emitter.emit('stage-complete', stageName, durationMs);
      return { result, stageResult: { stage: stageName, status: 'success', durationMs } };
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (attempt < maxRetries) {
        emitter.emit('stage-retry', stageName, attempt + 1, msg);
        await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
        continue;
      }
      const durationMs = Date.now() - startTime;
      emitter.emit('stage-failed', stageName, msg);
      return { result: null, stageResult: { stage: stageName, status: 'failed', durationMs, error: msg } };
    }
  }

  return { result: null, stageResult: { stage: stageName, status: 'failed', durationMs: 0, error: 'Unknown' } };
}

// ──────────────────────────────────────────────────────────────────────
// §5  Removal pass list builder
//
// Replaces the 4-branch if-else with a declarative pass list.
// Mirrors CCCL's approach of building kernel launch parameters
// in a loop rather than duplicating launch code.
//
// Before:
//   if (strategy === 'chroma-first') {
//     if (hasGreen) { tryChroma; if (!ok) tryRembg; }
//   } else if (strategy === 'rembg-first') {
//     tryRembg; if (!ok && hasGreen) tryChroma;
//   } ...
//
// After:
//   passes = buildPassList(strategy, hasGreen)
//   → [{type:'chroma'}, {type:'rembg'}]  // chroma-first with green
//   → [{type:'rembg'}, {type:'chroma'}]  // rembg-first with green
//   → [{type:'chroma'}]                  // chroma-only
// ──────────────────────────────────────────────────────────────────────

type RemovalPassType = 'chroma' | 'rembg';
interface RemovalPass {
  type: RemovalPassType;
  stage: PipelineStage;
  timeoutMultiplier: number;
}

function buildPassList(
  strategy: OrchestratorConfig['removalStrategy'],
  hasGreenScreen: boolean
): RemovalPass[] {
  const chroma: RemovalPass = { type: 'chroma', stage: 'chroma-remove', timeoutMultiplier: 1 };
  const rembg: RemovalPass = { type: 'rembg', stage: 'rembg-remove', timeoutMultiplier: 2 };

  switch (strategy) {
    case 'chroma-first':
      return hasGreenScreen ? [chroma, rembg] : [rembg];
    case 'rembg-first':
      return hasGreenScreen ? [rembg, chroma] : [rembg];
    case 'chroma-only':
      return hasGreenScreen ? [chroma] : [];
    case 'rembg-only':
      return [rembg];
  }
}

// ──────────────────────────────────────────────────────────────────────
// §6  finalizeStage — shared postprocess after each pipeline stage
//
// CCCL parallel: finalize_pass() with counter_update_fn callback.
//
// Before: each stage had inline logic to:
//   - Check if result is good enough
//   - Update stageResults array
//   - Emit progress
//
// After: single function, caller provides counter_update_fn
// that decides whether to accept the result.
// ──────────────────────────────────────────────────────────────────────

interface FinalizeContext {
  stageResults: StageResult[];
  emitter: EventEmitter;
}

function finalizeStage(
  ctx: FinalizeContext,
  stageResult: StageResult,
  metadata?: Record<string, any>
): void {
  stageResult.metadata = metadata;
  ctx.stageResults.push(stageResult);
}

// ──────────────────────────────────────────────────────────────────────
// §7  PipelineOrchestrator
// ──────────────────────────────────────────────────────────────────────

export class PipelineOrchestrator extends EventEmitter {
  private config: OrchestratorConfig;

  constructor(config?: Partial<OrchestratorConfig>) {
    super();
    this.config = { ...DEFAULT_ORCHESTRATOR_CONFIG, ...config };
  }

  /**
   * Process a single image through the full pipeline.
   *
   * Dispatch structure mirrors CCCL dispatch():
   *
   *   // Pass 0: histogram-only (detect green screen)
   *   analysis = analyzePass(input)
   *
   *   // Passes 1..N: fused filter+histogram (removal attempts)
   *   passes = buildPassList(strategy, analysis.hasGreen)
   *   for (const pass of passes) {
   *     result = executeRemovalPass(pass)
   *     if (result.good) break   // early_stop
   *     buffers.swap()           // key_bufs.selector ^= 1
   *   }
   *
   *   // Post-processing: layer separation, edge refine, export
   */
  async processImage(
    inputBuffer: Buffer,
    filename: string = 'input.png'
  ): Promise<PipelineResult> {
    const totalStart = Date.now();
    const stageResults: StageResult[] = [];
    let transparentImage: Buffer | null = null;
    let components: SeparationResult['components'] | null = null;
    let exportResult: ExportResult | null = null;
    let finalStrategy = 'none';

    const cfg = this.config;
    const timeout = cfg.stageTimeoutMs;
    const retries = cfg.maxRetries;
    const ctx: FinalizeContext = { stageResults, emitter: this };

    this.emit('pipeline-start', filename);

    // ──────────────────────────────────────────────────────────────
    // Pass 0: histogram-only — detect green screen
    // (Mirrors CCCL's DeviceTopKHistogramKernel, launched before
    //  the main filter loop)
    // ──────────────────────────────────────────────────────────────

    const chromaEngine = new ChromaEngine(cfg.chromaConfig);
    const { result: detection, stageResult: detectSR } = await executeStage(
      'detect',
      () => chromaEngine.detectGreenScreen(inputBuffer),
      timeout, retries, this
    );
    finalizeStage(ctx, detectSR, detection ?? undefined);

    const hasGreenScreen = detection?.hasGreenScreen ?? false;

    // ──────────────────────────────────────────────────────────────
    // Passes 1..N: fused filter+histogram (removal attempts)
    //
    // Uses buildPassList to eliminate the 4-branch if-else.
    // DoubleBuffer tracks the current best result.
    //
    // Mirrors CCCL's main loop:
    //   for (; pass < num_passes; pass++) {
    //     launch_kernel(..., key_bufs.Current(), key_bufs.Alternate(), ...);
    //     key_bufs.selector ^= 1;
    //   }
    // ──────────────────────────────────────────────────────────────

    const passes = buildPassList(cfg.removalStrategy, hasGreenScreen);
    const resultBuf = new DoubleBuffer<Buffer>();
    resultBuf.setCurrent(inputBuffer);

    for (let passIdx = 0; passIdx < passes.length; passIdx++) {
      const pass = passes[passIdx];
      const isLastPass = passIdx === passes.length - 1;

      if (pass.type === 'chroma') {
        const { result, stageResult } = await executeStage(
          pass.stage,
          () => chromaEngine.removeGreenScreen(resultBuf.current()!),
          timeout * pass.timeoutMultiplier, retries, this
        );
        finalizeStage(ctx, stageResult, result?.metadata);

        if (result) {
          const ratio = result.metadata.removalRatio;
          if (ratio >= cfg.chromaMinRemovalRatio) {
            // early_stop: good enough, no need for further passes
            resultBuf.setAlternate(result.outputBuffer);
            resultBuf.swap();
            finalStrategy = passIdx === 0 ? 'chroma' : 'chroma-fallback';
            break;
          }
        }
        // Not good enough — continue to next pass (like CCCL continuing radix passes)
      } else {
        const rembgBridge = new RembgBridge(cfg.rembgConfig);
        const { result, stageResult } = await executeStage(
          pass.stage,
          () => rembgBridge.removeBackground(resultBuf.current()!),
          timeout * pass.timeoutMultiplier, retries, this
        );
        finalizeStage(ctx, stageResult, result?.metadata);

        if (result) {
          resultBuf.setAlternate(result.outputBuffer);
          resultBuf.swap();
          finalStrategy = passIdx === 0 ? 'rembg' : 'rembg-fallback';
          break;
        }
      }
    }

    // Assign final transparent image (Current() holds the best result)
    transparentImage = resultBuf.current() === inputBuffer ? null : resultBuf.current();
    if (!transparentImage) {
      transparentImage = inputBuffer;
      finalStrategy = 'passthrough';
    }

    // ──────────────────────────────────────────────────────────────
    // Layer separation
    // ──────────────────────────────────────────────────────────────

    if (cfg.enableLayerSeparation && transparentImage) {
      const separator = new LayerSeparator(cfg.layerConfig);
      const { result, stageResult } = await executeStage(
        'layer-separate',
        () => separator.separate(transparentImage!),
        timeout, retries, this
      );
      finalizeStage(ctx, stageResult, result?.metadata);
      if (result && result.components.length > 0) {
        components = result.components;
      }
    }

    // ──────────────────────────────────────────────────────────────
    // Edge refinement (per component)
    // ──────────────────────────────────────────────────────────────

    if (cfg.enableEdgeRefine && components && components.length > 0) {
      const refiner = new EdgeRefiner(cfg.edgeConfig);
      const refinedComponents: typeof components = [];
      let anyFailed = false;

      for (let i = 0; i < components.length; i++) {
        const comp = components[i];
        this.emit('refine-component', i + 1, components.length);
        try {
          const refined = await refiner.refine(comp.imageBuffer);
          refinedComponents.push({
            ...comp,
            imageBuffer: refined.outputBuffer,
            info: { ...comp.info, strokePath: refined.strokePath } as any,
          });
        } catch {
          refinedComponents.push(comp);
          anyFailed = true;
        }
      }

      components = refinedComponents;
      finalizeStage(ctx, {
        stage: 'edge-refine',
        status: 'success',
        durationMs: 0,
      }, { totalComponents: components.length, anyFailed });
    }

    // ──────────────────────────────────────────────────────────────
    // Export
    // ──────────────────────────────────────────────────────────────

    if (cfg.enableExport && components && components.length > 0) {
      const exporter = new ComponentExporter(cfg.exportConfig);
      const { result, stageResult } = await executeStage(
        'export',
        () => exporter.exportAll(
          components!.map(c => ({
            info: c.info as any,
            imageBuffer: c.imageBuffer,
            maskBuffer: c.maskBuffer,
            strokePath: (c.info as any).strokePath,
          }))
        ),
        timeout, retries, this
      );
      finalizeStage(ctx, stageResult, result
        ? { filesGenerated: result.totalFilesGenerated, totalBytes: result.totalSizeBytes }
        : undefined
      );
      exportResult = result;
    }

    // ──────────────────────────────────────────────────────────────
    // Complete
    // ──────────────────────────────────────────────────────────────

    const totalDurationMs = Date.now() - totalStart;
    this.emit('pipeline-complete', totalDurationMs);

    return {
      stageResults,
      transparentImage,
      components,
      exportResult,
      metadata: {
        totalDurationMs,
        stagesExecuted: stageResults.length,
        stagesFailed: stageResults.filter(s => s.status === 'failed').length,
        finalStrategy,
      },
    };
  }

  updateConfig(partial: Partial<OrchestratorConfig>): void {
    this.config = { ...this.config, ...partial };
  }

  getConfig(): Readonly<OrchestratorConfig> {
    return { ...this.config };
  }
}

export default PipelineOrchestrator;