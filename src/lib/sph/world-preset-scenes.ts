/**
 * world-preset-scenes.ts — M760
 * 7 preset physics world configurations for demo/testing
 */

export interface PresetScene {
  name: string;
  description: string;
  setup: (addFluid: Function, addBody: Function, addEmitter: Function, W: number, H: number) => void;
}

export const PRESET_SCENES: Record<string, PresetScene> = {
  transformer_encoder: {
    name: 'Transformer Encoder',
    description: 'Standard 7-cell encoder block with dam break fluid',
    setup(addFluid, addBody, addEmitter, W, H) {
      const cells = [
        { id: 'input_embed', y: 0.08, sp: 2 },
        { id: 'pos_encode', y: 0.20, sp: 2 },
        { id: 'self_attn', y: 0.35, sp: 0 },
        { id: 'add_norm1', y: 0.48, sp: 3 },
        { id: 'ffn', y: 0.62, sp: 1 },
        { id: 'add_norm2', y: 0.75, sp: 3 },
        { id: 'output', y: 0.88, sp: 4 },
      ];
      cells.forEach(c => addBody(c.id, W * 0.5, H * c.y, 110, 32, c.sp, false));
      addFluid(10, H * 0.6, W * 0.25, H * 0.35, 3.2, 0);
      addFluid(W * 0.75, H * 0.6, W * 0.2, H * 0.3, 3.2, 5);
      addEmitter(W * 0.5, 10, 0, 1, 60, 2);
    },
  },
  attention_focus: {
    name: 'Attention Focus',
    description: 'Self-attention cell centered, radial fluid inflow',
    setup(addFluid, addBody, addEmitter, W, H) {
      addBody('self_attn', W / 2, H / 2, 140, 40, 0, true);
      for (let a = 0; a < 6; a++) {
        const angle = (a / 6) * Math.PI * 2;
        const r = Math.min(W, H) * 0.35;
        addEmitter(W / 2 + Math.cos(angle) * r, H / 2 + Math.sin(angle) * r, -Math.cos(angle), -Math.sin(angle), 40, a);
      }
    },
  },
  data_river: {
    name: 'Data River',
    description: 'Cells in a column, fluid flows top to bottom',
    setup(addFluid, addBody, addEmitter, W, H) {
      const names = ['embed', 'attn', 'norm1', 'ffn', 'norm2', 'proj', 'out'];
      names.forEach((n, i) => addBody(n, W / 2, 60 + i * ((H - 120) / 6), 100, 28, i % 7, false));
      addFluid(W * 0.3, 10, W * 0.4, 60, 3, 2);
      addEmitter(W / 2, 5, 0, 1, 120, 0);
    },
  },
  collision_test: {
    name: 'Collision Test',
    description: 'All cells drop from the top, collide with each other',
    setup(addFluid, addBody, addEmitter, W, H) {
      for (let i = 0; i < 12; i++) {
        addBody(`cell_${i}`, 50 + Math.random() * (W - 100), 30 + Math.random() * 100, 80 + Math.random() * 40, 28, i % 7, false);
      }
      addFluid(10, H - 120, W - 20, 100, 3, 1);
    },
  },
  qos_demo: {
    name: 'QoS Comparison',
    description: 'Split screen: left RELIABLE (viscous), right BEST_EFFORT (inviscid)',
    setup(addFluid, addBody, addEmitter, W, H) {
      addBody('wall', W / 2, H / 2, 4, H, 4, true);
      addFluid(10, H * 0.5, W * 0.4, H * 0.4, 3, 0);
      addFluid(W * 0.55, H * 0.5, W * 0.4, H * 0.4, 3, 5);
      addBody('left_cell', W * 0.25, H * 0.3, 80, 28, 0, false);
      addBody('right_cell', W * 0.75, H * 0.3, 80, 28, 1, false);
    },
  },
  empty_world: {
    name: 'Empty World',
    description: 'Blank canvas — add fluid and cells manually',
    setup() { /* intentionally empty */ },
  },
  stress_test: {
    name: 'Stress Test',
    description: '20000 particles + 20 rigid bodies',
    setup(addFluid, addBody, addEmitter, W, H) {
      for (let i = 0; i < 20; i++) {
        addBody(`stress_${i}`, 40 + Math.random() * (W - 80), 40 + Math.random() * (H - 80), 60 + Math.random() * 30, 24 + Math.random() * 10, i % 7, false);
      }
      addFluid(10, 10, W - 20, H * 0.7, 2.5, Math.floor(Math.random() * 7));
      addEmitter(W / 2, 5, 0, 1, 200, 3);
      addEmitter(10, H / 2, 1, 0, 150, 6);
    },
  },
};

export function getPresetNames(): string[] {
  return Object.keys(PRESET_SCENES);
}

export function setupPreset(name: string, addFluid: Function, addBody: Function, addEmitter: Function, W: number, H: number): void {
  const preset = PRESET_SCENES[name];
  if (preset) preset.setup(addFluid, addBody, addEmitter, W, H);
}
