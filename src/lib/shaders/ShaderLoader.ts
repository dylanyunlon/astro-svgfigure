import compiledRaw from './compiled.vs?raw';

// Standalone shader files — Vite ?raw imports
import pbrCellSurface from './pbr-cell-surface.frag?raw';
import voronoiMembrane from './voronoi-membrane.frag?raw';
import sdfSpeciesLib from './sdf-species-library.frag?raw';
import edgeSplineFrag from './edge-spline.frag?raw';
import edgeSplineVert from './edge-spline.vert?raw';
import msdfFrag from './msdf.frag?raw';
import msdfVert from './msdf.vert?raw';

const _cache: Map<string, string> = new Map();
let _parsed = false;

function parse(): void {
  if (_parsed) return;
  _parsed = true;

  // 1. Parse compiled.vs (AT production shaders)
  const parts = compiledRaw.split('{@}');
  for (let i = 1; i < parts.length - 1; i += 2) {
    const name = parts[i].trim();
    const content = parts[i + 1];
    _cache.set(name, content);
  }

  // 2. Register standalone shader files
  _cache.set('pbr-cell-surface.frag', pbrCellSurface);
  _cache.set('voronoi-membrane.frag', voronoiMembrane);
  _cache.set('sdf-species-library.frag', sdfSpeciesLib);
  _cache.set('edge-spline.frag', edgeSplineFrag);
  _cache.set('edge-spline.vert', edgeSplineVert);
  _cache.set('msdf.frag', msdfFrag);
  _cache.set('msdf.vert', msdfVert);

  // 3. Fallbacks for shaders referenced but only exist under different names
  //    bloom-luminosity.fs → BloomLuminosityPass.fs (AT name)
  //    bloom-blur.fs → UnrealBloomGaussian.fs
  //    bloom-composite.fs → UnrealBloomComposite.fs
  //    bloom-upsample.fs → HydraBloomPass.fs
  //    composite.fs → GlobalComposite.fs
  //    lighting.fs → lights.fs
  //    ShadowDepth.fs → ShadowDepth.glsl
  const aliases: [string, string][] = [
    ['bloom-luminosity.fs', 'BloomLuminosityPass.glsl'],
    ['bloom-blur.fs', 'UnrealBloomGaussian.glsl'],
    ['bloom-composite.fs', 'UnrealBloomComposite.glsl'],
    ['bloom-upsample.fs', 'HydraBloomPass.glsl'],
    ['bloomUpsample.fs', 'HydraBloomPass.glsl'],
    ['composite.fs', 'GlobalComposite.fs'],
    ['lighting.fs', 'lights.fs'],
    ['ShadowDepth.fs', 'ShadowDepth.glsl'],
  ];
  for (const [alias, target] of aliases) {
    if (!_cache.has(alias) && _cache.has(target)) {
      _cache.set(alias, _cache.get(target)!);
    }
  }
}

export function getShader(name: string): string {
  parse();
  const code = _cache.get(name);
  if (!code) {
    throw new Error(`[ShaderLoader] shader "${name}" not found in compiled.vs`);
  }
  return code;
}
