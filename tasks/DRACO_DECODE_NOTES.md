# M959: Draco Decode Verification — hexagon_gem.bin

## Package Used

`draco3dgltf` (v1.5.7) — installed via:
```
npm install draco3dgltf
```

`draco3d` is an alternative but `draco3dgltf` bundles a ready-to-use Node.js
decoder module and ships with a WASM backend, making it easier to use from
scripts without a build step.

## Custom .bin Wrapper Format (activetheory)

All geometry assets use a thin custom header before the raw Draco payload:

| Offset | Length | Value                         |
|--------|--------|-------------------------------|
| 0x00   | 2      | u16 file-type tag (0x38 0x33) |
| 0x02   | 8      | zero padding                  |
| 0x0A   | N      | UTF-8 JSON metadata           |
| 0x0A+N | rest  | raw Draco data (magic: DRACO) |

JSON metadata example:
```json
{"name":"hexagon_gem","type":0,"attributes":[["position",7],["normal",7],["uv",7]]}
```

The Draco data starts immediately after the closing `}` of the JSON.

## Decode Results — hexagon_gem.bin (501 bytes)

```
geometryType : TRIANGULAR_MESH
vertexCount  : 60
faceCount    : 32

Attributes
  position : ✓  (180 floats = 60 verts × 3)
  normal   : ✓  (180 floats = 60 verts × 3)
  uv       : ✓  (120 floats = 60 verts × 2)

First 3 vertices — position (x, y, z)
  v0: ( -0.650436,  0.375572,  -0.226756 )
  v1: ( -0.650436,  0.375572,  -0.226756 )
  v2: ( -0.650436,  0.375572,  -0.226756 )

First 3 vertices — normal (x, y, z)
  n0: ( 0.724861,  0.000000,  0.688895 )
  n1: ( 0.361324, -0.627391,  0.689801 )
  n2: ( 0.000000,  0.000000,  1.000000 )

First 3 vertices — uv (u, v)
  uv0: ( 0.092593,  0.363367 )
  uv1: ( 0.092593,  0.363367 )
  uv2: ( 0.092593,  0.363367 )
```

## Decode Script

`tasks/draco-decode-test.mjs` — run with:
```
node tasks/draco-decode-test.mjs
```
