#!/usr/bin/env node
// CommonJS module
/**
 * generate-species-glb.mjs — M1264: Generate GLB files for 5 cell species
 *
 * Each species encodes its Transformer algorithm semantics into 3D geometry:
 *   cil-eye:         radial attention rays + central pupil (multi-head attention)
 *   cil-bolt:        ReLU zigzag surface + energy core (FFN activation)
 *   cil-vector:      parallel groove capsule + direction arrows (embedding vectors)
 *   cil-plus:        cross junction + converging streams (residual add-norm)
 *   cil-arrow-right: streamlined arrow + flow ridges (output forward pass)
 *
 * Output: public/models/{species}.glb
 *
 * Usage: node tasks/generate-species-glb.mjs
 */

const THREE = require('three');
const { FileReader } = require('vblob');
global.FileReader = FileReader;
const { GLTFExporter } = require('three/examples/jsm/exporters/GLTFExporter.js');
const fs = require('fs');
const path = require('path');

const OUT_DIR = path.join(__dirname, '..', 'public', 'models');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ─── cil-eye: Attention sphere with radial rays + pupil ──────────────────────
function createCilEye() {
  const group = new THREE.Group();

  // Main sphere body (translucent blue membrane)
  const sphereGeo = new THREE.SphereGeometry(0.4, 32, 24);
  const sphereMat = new THREE.MeshStandardMaterial({
    color: 0x3F51B5, roughness: 0.4, metalness: 0.1, transparent: true, opacity: 0.85,
  });
  group.add(new THREE.Mesh(sphereGeo, sphereMat));

  // Central pupil (dark core)
  const pupilGeo = new THREE.SphereGeometry(0.08, 16, 12);
  const pupilMat = new THREE.MeshStandardMaterial({ color: 0x1A237E, roughness: 0.2, metalness: 0.6 });
  group.add(new THREE.Mesh(pupilGeo, pupilMat));

  // Iris ring
  const irisGeo = new THREE.TorusGeometry(0.12, 0.02, 8, 24);
  const irisMat = new THREE.MeshStandardMaterial({ color: 0x5C6BC0, roughness: 0.3, metalness: 0.4 });
  group.add(new THREE.Mesh(irisGeo, irisMat));

  // Radial attention rays — 8 thin cylinders radiating from center
  const numRays = 8;
  for (let i = 0; i < numRays; i++) {
    const angle = (Math.PI * 2 * i) / numRays;
    const rayGeo = new THREE.CylinderGeometry(0.008, 0.003, 0.35, 4);
    const rayMat = new THREE.MeshStandardMaterial({
      color: 0x7986CB, roughness: 0.5, metalness: 0.2,
      transparent: true, opacity: 0.6 - i * 0.03,
    });
    const ray = new THREE.Mesh(rayGeo, rayMat);
    ray.rotation.z = angle;
    ray.position.set(Math.cos(angle) * 0.2, Math.sin(angle) * 0.2, 0.05);
    group.add(ray);
  }

  // Lens bumps on sphere surface (4 positions, tetrahedral)
  const lensPositions = [
    [0, 0, 0.42], [0, 0, -0.42],
    [0.35, 0.2, 0], [-0.35, -0.2, 0],
  ];
  for (const [lx, ly, lz] of lensPositions) {
    const lensGeo = new THREE.SphereGeometry(0.06, 12, 8);
    const lensMat = new THREE.MeshStandardMaterial({
      color: 0x9FA8DA, roughness: 0.15, metalness: 0.7,
    });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.position.set(lx, ly, lz);
    group.add(lens);
  }

  return group;
}

// ─── cil-bolt: ReLU zigzag crystal + energy core ─────────────────────────────
function createCilBolt() {
  const group = new THREE.Group();

  // Diamond body
  const diamondGeo = new THREE.OctahedronGeometry(0.35, 0);
  const diamondMat = new THREE.MeshStandardMaterial({
    color: 0xF57C00, roughness: 0.25, metalness: 0.7,
  });
  const diamond = new THREE.Mesh(diamondGeo, diamondMat);
  diamond.scale.set(1, 1.3, 0.8);
  group.add(diamond);

  // ReLU zigzag lines on surface — the signature FFN activation shape
  // 3 zigzag strips wrapping around the crystal
  for (let strip = 0; strip < 3; strip++) {
    const points = [];
    const segments = 8;
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const x = (t - 0.5) * 0.7;
      // ReLU shape: flat for x<0, linear for x>0
      const reluY = t < 0.5 ? 0 : (t - 0.5) * 0.3;
      const z = strip * 0.12 - 0.12;
      points.push(new THREE.Vector3(x, reluY + 0.02, z + 0.36));
    }
    const zigzagGeo = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(points), 16, 0.008, 4, false
    );
    const zigzagMat = new THREE.MeshStandardMaterial({
      color: 0xFFB74D, roughness: 0.3, metalness: 0.5,
      emissive: 0xFF6F00, emissiveIntensity: 0.3,
    });
    group.add(new THREE.Mesh(zigzagGeo, zigzagMat));
  }

  // Energy core (glowing inner sphere)
  const coreGeo = new THREE.SphereGeometry(0.08, 12, 8);
  const coreMat = new THREE.MeshStandardMaterial({
    color: 0xFFE0B2, roughness: 0.1, metalness: 0.3,
    emissive: 0xFF9800, emissiveIntensity: 0.8,
  });
  group.add(new THREE.Mesh(coreGeo, coreMat));

  return group;
}

// ─── cil-vector: Capsule with parallel grooves + direction arrows ────────────
function createCilVector() {
  const group = new THREE.Group();

  // Capsule body (elongated)
  const capsuleGeo = new THREE.CapsuleGeometry(0.18, 0.5, 16, 24);
  const capsuleMat = new THREE.MeshStandardMaterial({
    color: 0x546E7A, roughness: 0.45, metalness: 0.3,
  });
  const capsule = new THREE.Mesh(capsuleGeo, capsuleMat);
  capsule.rotation.z = Math.PI / 2; // horizontal
  group.add(capsule);

  // Parallel groove lines along the body (embedding dimensions)
  const numGrooves = 6;
  for (let i = 0; i < numGrooves; i++) {
    const angle = (Math.PI * 2 * i) / numGrooves;
    const groovePoints = [];
    for (let t = 0; t <= 10; t++) {
      const x = (t / 10 - 0.5) * 0.6;
      const y = Math.cos(angle) * 0.19;
      const z = Math.sin(angle) * 0.19;
      groovePoints.push(new THREE.Vector3(x, y, z));
    }
    const grooveGeo = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(groovePoints), 8, 0.005, 3, false
    );
    const grooveMat = new THREE.MeshStandardMaterial({
      color: 0x78909C, roughness: 0.6, metalness: 0.2,
    });
    group.add(new THREE.Mesh(grooveGeo, grooveMat));
  }

  // Direction arrows (5 small arrows showing vector direction spread)
  for (let i = 0; i < 5; i++) {
    const spread = (i - 2) * 0.08;
    const arrowGeo = new THREE.ConeGeometry(0.02, 0.06, 6);
    const arrowMat = new THREE.MeshStandardMaterial({
      color: 0x2E7D32, roughness: 0.3, metalness: 0.5,
    });
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    arrow.rotation.z = -Math.PI / 2; // point right
    arrow.position.set(0.38, spread, 0);
    group.add(arrow);
  }

  return group;
}

// ─── cil-plus: Cross junction + converging streams ───────────────────────────
function createCilPlus() {
  const group = new THREE.Group();

  // Cross arms (X and Y)
  const armLen = 0.35, armW = 0.1, armD = 0.08;

  const xArmGeo = new THREE.BoxGeometry(armLen * 2, armW, armD);
  const xArmMat = new THREE.MeshStandardMaterial({
    color: 0xC62828, roughness: 0.4, metalness: 0.2,
  });
  group.add(new THREE.Mesh(xArmGeo, xArmMat));

  const yArmGeo = new THREE.BoxGeometry(armW, armLen * 2, armD);
  group.add(new THREE.Mesh(yArmGeo, xArmMat.clone()));

  // Center junction sphere (merge point)
  const junctionGeo = new THREE.SphereGeometry(0.07, 12, 8);
  const junctionMat = new THREE.MeshStandardMaterial({
    color: 0xEF5350, roughness: 0.2, metalness: 0.6,
    emissive: 0xC62828, emissiveIntensity: 0.3,
  });
  group.add(new THREE.Mesh(junctionGeo, junctionMat));

  // Converging dashed streams from 4 corners to center
  const corners = [[-1,-1], [1,-1], [-1,1], [1,1]];
  for (const [dx, dy] of corners) {
    const streamPoints = [];
    for (let t = 0; t <= 5; t++) {
      const frac = t / 5;
      streamPoints.push(new THREE.Vector3(
        dx * armLen * 1.1 * (1 - frac),
        dy * armLen * 1.1 * (1 - frac),
        0.05
      ));
    }
    const streamGeo = new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(streamPoints), 6, 0.006, 3, false
    );
    const streamMat = new THREE.MeshStandardMaterial({
      color: 0xEF9A9A, roughness: 0.5, metalness: 0.1,
      transparent: true, opacity: 0.5,
    });
    group.add(new THREE.Mesh(streamGeo, streamMat));
  }

  return group;
}

// ─── cil-arrow-right: Arrow with flow ridges ─────────────────────────────────
function createCilArrowRight() {
  const group = new THREE.Group();

  // Shaft (cylinder)
  const shaftGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.5, 12);
  const shaftMat = new THREE.MeshStandardMaterial({
    color: 0x455A64, roughness: 0.4, metalness: 0.3,
  });
  const shaft = new THREE.Mesh(shaftGeo, shaftMat);
  shaft.rotation.z = -Math.PI / 2;
  shaft.position.x = -0.05;
  group.add(shaft);

  // Arrowhead (cone)
  const headGeo = new THREE.ConeGeometry(0.18, 0.25, 8);
  const headMat = new THREE.MeshStandardMaterial({
    color: 0x455A64, roughness: 0.3, metalness: 0.4,
  });
  const head = new THREE.Mesh(headGeo, headMat);
  head.rotation.z = -Math.PI / 2;
  head.position.x = 0.33;
  group.add(head);

  // Flow ridges along shaft (3 ridges)
  for (let i = 0; i < 3; i++) {
    const ridgeGeo = new THREE.TorusGeometry(0.09, 0.01, 6, 16);
    const ridgeMat = new THREE.MeshStandardMaterial({
      color: 0x66BB6A, roughness: 0.3, metalness: 0.5,
      emissive: 0x2E7D32, emissiveIntensity: 0.2,
    });
    const ridge = new THREE.Mesh(ridgeGeo, ridgeMat);
    ridge.rotation.y = Math.PI / 2;
    ridge.position.x = -0.2 + i * 0.15;
    group.add(ridge);
  }

  return group;
}

// ─── Export all species ──────────────────────────────────────────────────────

const SPECIES = {
  'cil-eye': createCilEye,
  'cil-bolt': createCilBolt,
  'cil-vector': createCilVector,
  'cil-plus': createCilPlus,
  'cil-arrow-right': createCilArrowRight,
};

async function exportGLB(scene, outputPath) {
  return new Promise((resolve, reject) => {
    const exporter = new GLTFExporter();
    exporter.parse(
      scene,
      (result) => {
        const buffer = Buffer.from(result);
        fs.writeFileSync(outputPath, buffer);
        resolve(buffer.byteLength);
      },
      (error) => reject(error),
      { binary: true }
    );
  });
}

async function main() {
  console.log('Generating species GLB files...\n');

  for (const [species, createFn] of Object.entries(SPECIES)) {
    const scene = new THREE.Scene();
    const group = createFn();
    scene.add(group);

    const outPath = path.join(OUT_DIR, `${species}.glb`);
    try {
      const bytes = await exportGLB(scene, outPath);
      console.log(`  ✓ ${species}.glb  (${(bytes / 1024).toFixed(1)} KB)`);
    } catch (e) {
      console.error(`  ✗ ${species}: ${e.message}`);
    }
  }

  console.log(`\nDone. Files in ${OUT_DIR}/`);
}

main().catch(console.error);
