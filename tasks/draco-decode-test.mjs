/**
 * M959: Draco decode test — hexagon_gem.bin verified
 *
 * Custom .bin wrapper format (activetheory):
 *   [0x00–0x01]  2 bytes  — file-type tag (0x38 0x33 = "83")
 *   [0x02–0x09]  8 zero bytes
 *   [0x0A–...]   JSON metadata  {"name":..., "type":..., "attributes":...}
 *   [...–end]    raw Draco geometry (magic: "DRACO\x02\x02…")
 */

import { readFileSync } from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const draco3dgltf = require('draco3dgltf');

const BIN_PATH = new URL(
  '../upstream/activetheory-assets/geometry/hexagon_gem.bin',
  import.meta.url
).pathname;

// ── 1. Parse the custom .bin wrapper ────────────────────────────────────────
const raw        = readFileSync(BIN_PATH);
const dracoOffset = raw.indexOf(Buffer.from('DRACO'));
if (dracoOffset === -1) throw new Error('DRACO magic not found in file');

const headerBytes = raw.slice(0, dracoOffset);
const jsonStart   = headerBytes.indexOf(0x7B); // '{'
const jsonEnd     = headerBytes.lastIndexOf(0x7D) + 1;
const meta        = JSON.parse(headerBytes.slice(jsonStart, jsonEnd).toString('utf8'));
const dracoBuffer = raw.slice(dracoOffset);

console.log('── File metadata ───────────────────────────────────────────');
console.log('  name      :', meta.name);
console.log('  type      :', meta.type);
console.log('  attributes:', JSON.stringify(meta.attributes));
console.log('  total bytes:', raw.length);
console.log('  header bytes:', dracoOffset, '  draco bytes:', dracoBuffer.length);

// ── 2. Initialise Draco decoder ──────────────────────────────────────────────
const decoderModule = await draco3dgltf.createDecoderModule({});
const decoder       = new decoderModule.Decoder();

// ── 3. Decode ────────────────────────────────────────────────────────────────
const dbuf = new decoderModule.DecoderBuffer();
dbuf.Init(
  new Int8Array(dracoBuffer.buffer, dracoBuffer.byteOffset, dracoBuffer.byteLength),
  dracoBuffer.byteLength
);

const geoType = decoder.GetEncodedGeometryType(dbuf);
let geo, status;
if (geoType === decoderModule.TRIANGULAR_MESH) {
  geo    = new decoderModule.Mesh();
  status = decoder.DecodeBufferToMesh(dbuf, geo);
} else if (geoType === decoderModule.POINT_CLOUD) {
  geo    = new decoderModule.PointCloud();
  status = decoder.DecodeBufferToPointCloud(dbuf, geo);
} else {
  throw new Error('Unknown geometry type: ' + geoType);
}
decoderModule.destroy(dbuf);

if (!status.ok()) throw new Error('Draco decode failed: ' + status.error_msg());

const vertexCount = geo.num_points();
const faceCount   = (geoType === decoderModule.TRIANGULAR_MESH) ? geo.num_faces() : 0;
const geoTypeName = geoType === decoderModule.TRIANGULAR_MESH ? 'TRIANGULAR_MESH' : 'POINT_CLOUD';

console.log('\n── Decoded geometry ────────────────────────────────────────');
console.log('  geometryType :', geoTypeName);
console.log('  vertexCount  :', vertexCount);
console.log('  faceCount    :', faceCount);

// ── 4. Helper — extract a float attribute ────────────────────────────────────
function extractFloatAttr(dracoAttrType, components) {
  const attrId = decoder.GetAttributeId(geo, dracoAttrType);
  if (attrId < 0) return null;
  const attr = decoder.GetAttribute(geo, attrId);
  const tmp  = new decoderModule.DracoFloat32Array();
  decoder.GetAttributeFloatForAllPoints(geo, attr, tmp);
  const arr = new Float32Array(vertexCount * components);
  for (let i = 0; i < arr.length; i++) arr[i] = tmp.GetValue(i);
  decoderModule.destroy(tmp);
  return arr;
}

// ── 5. Extract position / normal / uv ────────────────────────────────────────
const posArr    = extractFloatAttr(decoderModule.POSITION,  3);
const normalArr = extractFloatAttr(decoderModule.NORMAL,    3);
const uvArr     = extractFloatAttr(decoderModule.TEX_COORD, 2);

console.log('\n── Attributes ──────────────────────────────────────────────');
console.log('  has position :', posArr    !== null, posArr    ? `(${posArr.length} floats = ${posArr.length/3} verts×3)` : '');
console.log('  has normal   :', normalArr !== null, normalArr ? `(${normalArr.length} floats = ${normalArr.length/3} verts×3)` : '');
console.log('  has uv       :', uvArr     !== null, uvArr     ? `(${uvArr.length} floats = ${uvArr.length/2} verts×2)` : '');

// ── 6. Print first 3 vertices ────────────────────────────────────────────────
if (posArr) {
  console.log('\n── First 3 vertices — position (x, y, z) ───────────────────');
  for (let i = 0; i < Math.min(3, vertexCount); i++) {
    const x = posArr[i*3+0].toFixed(6);
    const y = posArr[i*3+1].toFixed(6);
    const z = posArr[i*3+2].toFixed(6);
    console.log(`  v${i}: ( ${x},  ${y},  ${z} )`);
  }
}

if (normalArr) {
  console.log('\n── First 3 vertices — normal (x, y, z) ─────────────────────');
  for (let i = 0; i < Math.min(3, vertexCount); i++) {
    const x = normalArr[i*3+0].toFixed(6);
    const y = normalArr[i*3+1].toFixed(6);
    const z = normalArr[i*3+2].toFixed(6);
    console.log(`  n${i}: ( ${x},  ${y},  ${z} )`);
  }
}

if (uvArr) {
  console.log('\n── First 3 vertices — uv (u, v) ─────────────────────────────');
  for (let i = 0; i < Math.min(3, vertexCount); i++) {
    const u = uvArr[i*2+0].toFixed(6);
    const v = uvArr[i*2+1].toFixed(6);
    console.log(`  uv${i}: ( ${u},  ${v} )`);
  }
}

// ── 7. Cleanup ────────────────────────────────────────────────────────────────
decoderModule.destroy(decoder);
decoderModule.destroy(geo);

console.log('\n✓  hexagon_gem.bin decoded successfully');
