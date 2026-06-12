/**
 * GET /api/cells — Live cell status endpoint
 *
 * Reads channels/cell/*/status.json from the project root and
 * returns a CellDescriptor[] JSON array.
 *
 * File layout expected:
 *   channels/cell/<cell_id>/status.json  →  CellDescriptor
 *
 * Used by pixi-cell-renderer.ts pollCellChannels() every 500ms.
 *
 * GitHub 背书: withastro/astro (API Routes, SSR)
 */
import type { APIRoute } from 'astro';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

export const prerender = false;

// Root of the project — works for both dev and build (output dir shares cwd)
const CHANNELS_ROOT = join(process.cwd(), 'channels', 'cell');

export const GET: APIRoute = async () => {
  const cells: unknown[] = [];

  try {
    let entries: string[];
    try {
      entries = await readdir(CHANNELS_ROOT);
    } catch {
      // Directory doesn't exist yet — return empty array (no cells running)
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
        },
      });
    }

    await Promise.all(
      entries.map(async (entry) => {
        const statusPath = join(CHANNELS_ROOT, entry, 'status.json');
        try {
          const raw = await readFile(statusPath, 'utf-8');
          const descriptor = JSON.parse(raw);
          // Basic shape guard — must have cell_id and bbox
          if (
            descriptor &&
            typeof descriptor.cell_id === 'string' &&
            descriptor.bbox &&
            typeof descriptor.bbox.x === 'number'
          ) {
            cells.push(descriptor);
          }
        } catch {
          // Missing or malformed status.json — skip this cell silently
        }
      }),
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  return new Response(JSON.stringify(cells), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
    },
  });
};
