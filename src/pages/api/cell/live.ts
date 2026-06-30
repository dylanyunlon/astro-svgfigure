/**
 * GET /api/cell/live — SSE stream of live cell agent decisions
 *
 * When first connected, initializes the cell spawner (if not already running).
 * Then streams cell decisions as they happen:
 *   event: cell_decision
 *   data: { cell_id, action, params, reasoning, tick, energy }
 *
 * The GPU render loop in the browser listens to this and updates cell positions,
 * morphs, signals in real-time.
 */

import type { APIRoute } from 'astro';
import { init, getAgentStates, getCellDecision } from '../../lib/cell-spawner';

export const prerender = false;

let initialized = false;

export const GET: APIRoute = async ({ request }) => {
  // Initialize spawner on first connection
  if (!initialized) {
    try {
      init();
      initialized = true;
    } catch (e) {
      console.error('[cell/live] spawner init failed:', e);
    }
  }

  // Set up SSE stream
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      // Send initial state
      const states = getAgentStates();
      controller.enqueue(encoder.encode(
        `event: init\ndata: ${JSON.stringify({ cells: states.length, timestamp: Date.now() })}\n\n`
      ));

      // Poll for cell decisions and stream them
      const interval = setInterval(() => {
        try {
          const states = getAgentStates();
          for (const agent of states) {
            if (agent.lastAction !== 'none') {
              const decision = getCellDecision(agent.cellId);
              if (decision && decision.timestamp > Date.now() - 10000) {
                controller.enqueue(encoder.encode(
                  `event: cell_decision\ndata: ${JSON.stringify({
                    cell_id: agent.cellId,
                    species: agent.species,
                    action: decision.action,
                    params: decision.params,
                    reasoning: (decision as any).reasoning ?? '',
                    tick: agent.tick,
                    energy: agent.energy,
                  })}\n\n`
                ));
              }
            }
          }
        } catch (e) {
          // stream may be closed
        }
      }, 2000);

      // Clean up on disconnect
      request.signal.addEventListener('abort', () => {
        clearInterval(interval);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
};
