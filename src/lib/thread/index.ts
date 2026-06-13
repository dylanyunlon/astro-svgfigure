/**
 * src/lib/thread/index.ts — Thread module public API.
 *
 * Re-exports the Thread class and worker type contracts so callers can import
 * from a single path: `import { Thread } from '$lib/thread'`.
 */

export { Thread } from './Thread';

export type {
  CellParams,
  CellGeomInput,
  CellGeomResult,
  CellGeomOutput,
  CellAABBOutput,
  CellOutlineOutput,
} from './workers/geometry-worker';
