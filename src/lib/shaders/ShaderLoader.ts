import compiledRaw from './compiled.vs?raw';

const _cache: Map<string, string> = new Map();
let _parsed = false;

function parse(): void {
  if (_parsed) return;
  _parsed = true;

  const parts = compiledRaw.split('{@}');
  // parts layout: [preamble, name1, content1, name2, content2, ...]
  for (let i = 1; i < parts.length - 1; i += 2) {
    const name = parts[i].trim();
    const content = parts[i + 1];
    _cache.set(name, content);
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
