/**
 * cell-spawner.ts — Runtime Claude Code cell agents
 *
 * When the Astro server starts, this module spawns a Claude Code process
 * for each cell. Each cell agent:
 *   1. Reads its channel (channels/cell/{id}/)
 *   2. Perceives neighbors via channels/physics/
 *   3. Makes decisions (move, morph, divide, signal)
 *   4. Writes decisions to channels/cell/{id}/out.json
 *   5. GPU render loop reads out.json and updates rendering
 *   6. Loop continues
 *
 * Architecture:
 *   bun dev → Astro server starts
 *     → cell-spawner.init() called from API route or server middleware
 *     → for each cell: spawn claude CLI process via proxy
 *     → cell agent runs in agentic loop (read → decide → write → repeat)
 *     → SSE pushes cell updates to browser
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellAgent {
  cellId: string;
  species: string;
  process: ChildProcess | null;
  status: 'idle' | 'thinking' | 'acting' | 'dead';
  lastAction: string;
  energy: number;
  tick: number;
}

export interface CellDecision {
  action: 'move' | 'morph' | 'signal' | 'divide' | 'idle' | 'die';
  params: Record<string, unknown>;
  timestamp: number;
}

// ── Constants ────────────────────────────────────────────────────────────────

const CHANNELS_DIR = join(process.cwd(), 'channels');
const CELL_DIR = join(CHANNELS_DIR, 'cell');
const PHYSICS_DIR = join(CHANNELS_DIR, 'physics');

const PROXY_URL = 'http://127.0.0.1:19876';
const PROXY_KEY = 'sk-ant-proxy';

// How often each cell "thinks" (ms)
const CELL_TICK_INTERVAL = 5000;

// Max concurrent cell agents
const MAX_CONCURRENT = 5;

// ── Cell Agent Registry ──────────────────────────────────────────────────────

const agents = new Map<string, CellAgent>();
const tickQueue: string[] = [];
let spawnerRunning = false;

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Initialize cell spawner. Call once when server starts.
 * Reads composite_params.json, creates agents for each cell.
 */
export function init(): void {
  if (spawnerRunning) return;
  spawnerRunning = true;

  console.log('[CellSpawner] Initializing...');

  // Read all cells
  const paramsPath = join(CHANNELS_DIR, 'composite_params.json');
  if (!existsSync(paramsPath)) {
    console.warn('[CellSpawner] composite_params.json not found');
    return;
  }

  const params = JSON.parse(readFileSync(paramsPath, 'utf-8'));
  const cells = params.cells ?? {};

  for (const [cellId, cellData] of Object.entries(cells)) {
    const cd = cellData as Record<string, unknown>;
    const sp = (cd.agent_params as any)?.species_params ?? {};
    const species = sp.species ?? 'unknown';

    agents.set(cellId, {
      cellId,
      species,
      process: null,
      status: 'idle',
      lastAction: 'none',
      energy: 1.0,
      tick: 0,
    });
  }

  console.log(`[CellSpawner] ${agents.size} cell agents registered`);

  // Start the tick loop
  setInterval(tickLoop, CELL_TICK_INTERVAL);
  console.log(`[CellSpawner] Tick loop started (${CELL_TICK_INTERVAL}ms interval)`);
}

/**
 * Get all cell agent states (for SSE broadcast).
 */
export function getAgentStates(): CellAgent[] {
  return [...agents.values()].map(a => ({
    ...a,
    process: null, // don't serialize process
  }));
}

/**
 * Get a specific cell's latest decision.
 */
export function getCellDecision(cellId: string): CellDecision | null {
  const outPath = join(CELL_DIR, cellId, 'out.json');
  if (!existsSync(outPath)) return null;
  try {
    return JSON.parse(readFileSync(outPath, 'utf-8'));
  } catch {
    return null;
  }
}

// ── Tick Loop ────────────────────────────────────────────────────────────────

function tickLoop(): void {
  // Find cells that are idle and ready for a tick
  const idle = [...agents.values()].filter(a => a.status === 'idle');
  
  // Prioritize cells with higher energy (more active)
  idle.sort((a, b) => b.energy - a.energy);

  // Queue up to MAX_CONCURRENT cells for thinking
  let spawned = 0;
  const active = [...agents.values()].filter(a => a.status === 'thinking' || a.status === 'acting');

  for (const agent of idle) {
    if (active.length + spawned >= MAX_CONCURRENT) break;
    spawnCellTick(agent);
    spawned++;
  }
}

// ── Cell Tick (one round of perception → decision → action) ──────────────────

function spawnCellTick(agent: CellAgent): void {
  agent.status = 'thinking';
  agent.tick++;

  // Build the cell's perception prompt
  const perception = buildPerception(agent);

  const prompt = `你是细胞 ${agent.cellId}，species: ${agent.species}，tick: ${agent.tick}，energy: ${agent.energy.toFixed(2)}。

${perception}

基于你的感知，做一个决策。输出一个 JSON 到标准输出（不要 markdown fence）：
{
  "action": "move"|"morph"|"signal"|"divide"|"idle"|"die",
  "params": {
    // move: {"dx": number, "dy": number}
    // morph: {"scale": number, "rotation": number, "deform": [...]}
    // signal: {"type": "attract"|"repel"|"alert", "strength": 0-1, "radius": number}
    // divide: {"direction": [dx, dy], "energy_split": 0.5}
    // idle: {}
    // die: {"reason": "..."}
  },
  "reasoning": "一句话解释为什么做这个决策"
}`;

  // Use claude CLI in non-interactive mode
  const env = {
    ...process.env,
    ANTHROPIC_BASE_URL: PROXY_URL,
    ANTHROPIC_API_KEY: PROXY_KEY,
    DISABLE_AUTOUPDATER: '1',
  };

  const child = spawn('claude', [
    '-p', prompt,
    '--model', 'claude-sonnet-4-6',
    '--max-turns', '1',
  ], {
    env,
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  agent.process = child;
  agent.status = 'acting';

  let stdout = '';
  child.stdout?.on('data', (chunk: Buffer) => {
    stdout += chunk.toString();
  });

  child.on('close', (code: number | null) => {
    agent.process = null;

    try {
      // Extract JSON from output
      const jsonMatch = stdout.match(/\{[\s\S]*"action"[\s\S]*\}/);
      if (jsonMatch) {
        const decision: CellDecision = {
          ...JSON.parse(jsonMatch[0]),
          timestamp: Date.now(),
        };

        // Write decision to channel
        const outDir = join(CELL_DIR, agent.cellId);
        if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
        writeFileSync(join(outDir, 'out.json'), JSON.stringify(decision, null, 2));

        // Apply energy cost
        agent.energy -= 0.02; // thinking costs energy
        if (decision.action === 'move') agent.energy -= 0.01;
        if (decision.action === 'divide') agent.energy -= 0.4;
        if (decision.action === 'die') agent.status = 'dead';

        // Regenerate
        agent.energy = Math.min(1.0, agent.energy + 0.015);
        if (agent.energy <= 0) agent.status = 'dead';

        agent.lastAction = decision.action;
        console.log(`[Cell:${agent.cellId}] tick=${agent.tick} action=${decision.action} energy=${agent.energy.toFixed(2)}`);
      } else {
        console.warn(`[Cell:${agent.cellId}] no valid JSON in output`);
      }
    } catch (e) {
      console.warn(`[Cell:${agent.cellId}] decision parse error:`, e);
    }

    if (agent.status !== 'dead') {
      agent.status = 'idle';
    }
  });

  child.on('error', (err: Error) => {
    console.warn(`[Cell:${agent.cellId}] process error:`, err.message);
    agent.process = null;
    agent.status = 'idle';
  });

  // Timeout — kill if thinking too long
  setTimeout(() => {
    if (child.exitCode === null) {
      child.kill('SIGTERM');
      agent.status = 'idle';
    }
  }, 60000);
}

// ── Perception Builder ───────────────────────────────────────────────────────

function buildPerception(agent: CellAgent): string {
  const parts: string[] = [];

  // Read own channel
  const cellDir = join(CELL_DIR, agent.cellId);
  if (existsSync(join(cellDir, 'params.json'))) {
    try {
      const params = JSON.parse(readFileSync(join(cellDir, 'params.json'), 'utf-8'));
      parts.push(`你的参数: ${JSON.stringify(params).slice(0, 500)}`);
    } catch {}
  }

  if (existsSync(join(cellDir, 'bbox.json'))) {
    try {
      const bbox = JSON.parse(readFileSync(join(cellDir, 'bbox.json'), 'utf-8'));
      parts.push(`你的位置: x=${bbox.x}, y=${bbox.y}, w=${bbox.w}, h=${bbox.h}`);
    } catch {}
  }

  // Read interaction matrix to know relationships
  const matrixPath = join(PHYSICS_DIR, 'species_interaction_matrix.json');
  if (existsSync(matrixPath)) {
    try {
      const matrix = JSON.parse(readFileSync(matrixPath, 'utf-8'));
      const myRow = matrix.matrix?.[agent.species];
      if (myRow) {
        parts.push(`你对其他species的态度: ${JSON.stringify(myRow)}`);
      }
    } catch {}
  }

  // Read nearby cells (simplified — check a few neighbors)
  const allCells = [...agents.values()];
  const neighbors = allCells
    .filter(a => a.cellId !== agent.cellId && a.status !== 'dead')
    .slice(0, 6)
    .map(a => `${a.cellId}(${a.species}, energy=${a.energy.toFixed(2)}, last=${a.lastAction})`);

  if (neighbors.length > 0) {
    parts.push(`附近的细胞: ${neighbors.join(', ')}`);
  }

  // Force field
  const ffPath = join(PHYSICS_DIR, 'force_field.json');
  if (existsSync(ffPath)) {
    try {
      const ff = JSON.parse(readFileSync(ffPath, 'utf-8'));
      const myForce = ff[agent.cellId];
      if (myForce) {
        parts.push(`作用在你身上的力: dx=${myForce.dx}, dy=${myForce.dy}`);
      }
    } catch {}
  }

  return parts.join('\n');
}

// ── Shutdown ─────────────────────────────────────────────────────────────────

export function shutdown(): void {
  spawnerRunning = false;
  for (const agent of agents.values()) {
    if (agent.process) {
      agent.process.kill('SIGTERM');
      agent.process = null;
    }
  }
  console.log('[CellSpawner] Shutdown complete');
}
