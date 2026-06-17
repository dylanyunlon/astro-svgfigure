import { Assets } from 'pixi.js';

export interface CompositeParams {
  cells: any[];
  edges: any[];
  canvas: { width: number; height: number };
}

export async function loadCellAssets(): Promise<CompositeParams> {
  const d = await (await fetch("/api/cells")).json();
  for (const c of d.cells) {
    if (c.msdf_path)
      try {
        c.msdfTexture = await Assets.load(c.msdf_path);
      } catch (e) {}
  }
  return d;
}
