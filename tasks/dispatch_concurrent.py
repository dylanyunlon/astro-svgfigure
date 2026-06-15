#!/usr/bin/env python3
"""
dispatch_concurrent.py — Manager script for 60 concurrent xiaodi cells.

Key design (from owner's critique):
  60 concurrent xiaodi don't "notify neighbors" — they publish/subscribe
  through Apollo channel IDs.  Convergence comes from multi-round epoch
  loops where each cell:
    1. READS its subscribed channels (SubscriptionTable.get_subscriptions)
    2. Computes its params (sub-Claude with web search)
    3. PUBLISHES to its output channel (SubscriptionTable.publish)
    4. Only cells that SUBSCRIBED to that channel see the update

  This is NOT fan-out-to-all.  This IS targeted pub/sub.

Usage:
    # From repo root
    PYTHONPATH=. python3 tasks/dispatch_concurrent.py [--topology TRANSFORMER]
    PYTHONPATH=. python3 tasks/dispatch_concurrent.py --data path/to/data.json

Architecture:
    Manager (this script)
      → reads topology → builds SubscriptionTable
      → for each epoch:
          → dispatch up to 60 cells concurrently (ThreadPoolExecutor)
          → each cell = 1 sub-Claude API call OR local proc
          → cells publish results → only subscribers notified
          → physics convergence check
          → if not converged → next epoch
"""

import json
import os
import sys
import time
import threading
import concurrent.futures
from typing import Dict, List, Optional, Set

# Ensure repo root on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from channels.data.subscription_table import SubscriptionTable
from channels.data.notifier import DataNotifier
from channels.data.data_dispatcher import DataDispatcher

CHANNELS_DIR = os.path.join(REPO_ROOT, "channels")

# ── Configuration ────────────────────────────────────────────────────────────

MAX_CONCURRENT = 60          # Max parallel xiaodi
MAX_EPOCHS = 10              # Convergence loop limit
CONVERGENCE_THRESHOLD = 0.5  # px delta for convergence
INTER_CELL_DELAY = 0.1       # seconds between dispatches (rate limit)

# ── Sub-Claude dispatch modes ────────────────────────────────────────────────

DISPATCH_MODE = os.environ.get("ASTRO_DISPATCH_MODE", "local")
# "local"  → run cell_component.proc() in-process
# "api"    → dispatch to claude.hk.cn (requires cookie config)


def dispatch_cell_local(cell_id: str, table: SubscriptionTable) -> dict:
    """
    Local dispatch: run cell proc in-process.
    Reads subscribed channels, computes, publishes output.
    """
    from channels.cell_component import proc as cell_proc

    # 1. Read all subscribed channels
    subs = table.get_subscriptions(cell_id)
    inputs = {}
    for ch in subs:
        ch_path = os.path.join(CHANNELS_DIR, ch)
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                inputs[ch] = json.load(f)

    # 2. Run cell proc
    result = cell_proc(cell_id, inputs)

    # 3. Publish to output channel
    out_ch = f"cell/{cell_id}/out.json"
    notified = table.publish(out_ch, result)

    return {
        "cell_id": cell_id,
        "subscribed_from": list(subs),
        "published_to": out_ch,
        "notified_cells": notified,
        "result_keys": list(result.keys()) if isinstance(result, dict) else [],
    }


def dispatch_cell_api(cell_id: str, table: SubscriptionTable,
                      server_url: str) -> dict:
    """
    API dispatch: send to claude.hk.cn sub-Claude.
    The sub-Claude has web search and code execution.
    It researches its academic concept and returns params.
    """
    import urllib.request
    import urllib.error

    # Build the cell's context from its subscriptions
    subs = table.get_subscriptions(cell_id)
    context = {}
    for ch in subs:
        ch_path = os.path.join(CHANNELS_DIR, ch)
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                context[ch] = json.load(f)

    # Read cell skeleton for species info
    skel_path = os.path.join(CHANNELS_DIR, "skeleton", "cell", f"{cell_id}.json")
    species = "cil-code"
    label = cell_id
    if os.path.exists(skel_path):
        with open(skel_path) as f:
            skel = json.load(f)
            species = skel.get("species", species)
            label = skel.get("label", label)

    # Construct prompt for sub-Claude
    prompt = f"""You are xiaodi cell "{cell_id}" (species: {species}, label: "{label}").

Your SUBSCRIBED channels (what you can read):
{json.dumps(list(subs), indent=2)}

Current channel data:
{json.dumps(context, indent=2)}

Task:
1. Web search for "{label} diagram visualization" to find academic visual conventions
2. Based on your species ({species}) and the channel data, compute your visual params
3. Output ONLY valid JSON with keys: bbox, opacity, species_params

The bbox must respect the force_field constraints from your subscribed physics channels.
Do NOT broadcast to all cells. Your output goes ONLY to channel "cell/{cell_id}/out.json"
and ONLY cells that subscribed to it will see your result.
"""

    payload = {
        "cell_id": cell_id,
        "prompt": prompt,
        "subscriptions": list(subs),
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{server_url}/api/cell/dispatch",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())

        # Publish result through SubscriptionTable
        out_ch = f"cell/{cell_id}/out.json"
        notified = table.publish(out_ch, result)

        return {
            "cell_id": cell_id,
            "subscribed_from": list(subs),
            "published_to": out_ch,
            "notified_cells": notified,
            "status": "ok",
        }
    except Exception as e:
        return {
            "cell_id": cell_id,
            "status": "error",
            "error": str(e),
        }


# ── Epoch orchestration ─────────────────────────────────────────────────────

def run_concurrent_epoch(cell_ids: List[str], table: SubscriptionTable,
                         epoch: int, server_url: Optional[str] = None) -> dict:
    """
    Run one epoch with up to 60 concurrent cells.

    Unlike the old neighbor-broadcast model:
      - Each cell only reads its SUBSCRIBED channels
      - Each cell only publishes to its OWN output channel
      - Only cells that SUBSCRIBED to that output get notified
      - The physics engine collects ALL outputs and publishes force_field
        (a shared channel that everyone subscribes to)
    """
    print(f"\n{'─'*60}")
    print(f"  EPOCH {epoch}  ({len(cell_ids)} cells, max_concurrent={MAX_CONCURRENT})")
    print(f"{'─'*60}")

    results = {}
    dispatch_fn = dispatch_cell_local if DISPATCH_MODE == "local" else dispatch_cell_api

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {}
        for i, cell_id in enumerate(cell_ids):
            if DISPATCH_MODE == "api" and server_url:
                future = executor.submit(dispatch_cell_api, cell_id, table, server_url)
            else:
                future = executor.submit(dispatch_cell_local, cell_id, table)
            futures[future] = cell_id

            # Stagger to avoid rate limits
            if i < len(cell_ids) - 1:
                time.sleep(INTER_CELL_DELAY)

        for future in concurrent.futures.as_completed(futures):
            cell_id = futures[future]
            try:
                result = future.result(timeout=120)
                results[cell_id] = result
                notified = result.get("notified_cells", [])
                print(f"  [✓] {cell_id} → published, notified: {notified}")
            except Exception as e:
                results[cell_id] = {"cell_id": cell_id, "status": "error", "error": str(e)}
                print(f"  [✗] {cell_id} → {e}")

    # Print routing summary
    print(f"\n  Routing summary (epoch {epoch}):")
    total_notifications = 0
    for cell_id, result in results.items():
        n = len(result.get("notified_cells", []))
        total_notifications += n
    print(f"  Total notifications: {total_notifications}")
    print(f"  Avg notifications/cell: {total_notifications/max(1,len(cell_ids)):.1f}")

    return results


def check_convergence(results: dict) -> Tuple:
    """Check if all cells have converged (bbox delta < threshold)."""
    max_delta = 0.0
    for cell_id, result in results.items():
        if isinstance(result, dict) and "result_keys" in result:
            # Read the cell's output to check bbox delta
            out_path = os.path.join(CHANNELS_DIR, f"cell/{cell_id}/out.json")
            if os.path.exists(out_path):
                with open(out_path) as f:
                    out = json.load(f)
                delta = out.get("bbox_delta", 0)
                if isinstance(delta, dict):
                    delta = max(abs(delta.get("dx", 0)), abs(delta.get("dy", 0)))
                max_delta = max(max_delta, delta)

    converged = max_delta < CONVERGENCE_THRESHOLD
    return converged, max_delta


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concurrent xiaodi dispatch")
    parser.add_argument("--topology", default="TRANSFORMER", help="Topology name")
    parser.add_argument("--data", default=None, help="Path to structured data JSON")
    parser.add_argument("--server", default=None, help="Server URL for API dispatch")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    global MAX_CONCURRENT, MAX_EPOCHS
    MAX_CONCURRENT = args.max_concurrent
    MAX_EPOCHS = args.max_epochs

    # 1. Build subscription table from topology
    table = SubscriptionTable.instance()

    # Generate skeleton if needed
    os.chdir(CHANNELS_DIR)
    sys.path.insert(0, CHANNELS_DIR)
    from topology_to_skeleton import (
        parse_examples_ts, generate_skeleton, EXAMPLES_TS, SKELETON_DIR
    )

    if args.data:
        from topology_to_skeleton import from_structured_data
        cells, edges = from_structured_data(args.data)
    else:
        examples = parse_examples_ts(EXAMPLES_TS)
        topo = args.topology.upper()
        if topo in examples:
            cells = generate_skeleton(examples[topo], topo)
            print(f"[topology] Generated {len(cells)} cells for {topo}")

    # Load subscription table from topology.json
    table.load_from_topology()

    # Get cell IDs from skeleton
    skel_dir = os.path.join(CHANNELS_DIR, "skeleton", "cell")
    cell_ids = []
    if os.path.isdir(skel_dir):
        for f in os.listdir(skel_dir):
            if f.endswith(".json"):
                cell_ids.append(f[:-5])

    if not cell_ids:
        print("No cells found in skeleton. Exiting.")
        return

    # Print routing table
    routing = table.dump_routing_table()
    print(f"\n{'='*60}")
    print(f"  SUBSCRIPTION TABLE — {len(cell_ids)} cells")
    print(f"{'='*60}")
    for cell_id in sorted(cell_ids):
        cell_info = routing["cells"].get(cell_id, {})
        subs = cell_info.get("subscribes_to", [])
        pubs = cell_info.get("publishes_to", [])
        # Count only non-physics subscriptions (direct cell-to-cell)
        direct_subs = [s for s in subs if s.startswith("cell/")]
        print(f"  {cell_id}: reads {len(direct_subs)} cell channels + {len(subs)-len(direct_subs)} shared, publishes {len(pubs)}")

    # 2. Run epoch loop
    print(f"\n{'='*60}")
    print(f"  CONCURRENT DISPATCH — max {MAX_CONCURRENT} parallel, {MAX_EPOCHS} max epochs")
    print(f"  Mode: {DISPATCH_MODE}")
    print(f"{'='*60}")

    for epoch in range(MAX_EPOCHS):
        results = run_concurrent_epoch(cell_ids, table, epoch, args.server)

        converged, max_delta = check_convergence(results)
        print(f"\n  Convergence check: max_delta={max_delta:.2f}px "
              f"threshold={CONVERGENCE_THRESHOLD}px "
              f"{'✓ CONVERGED' if converged else '✗ continuing'}")

        if converged and epoch >= 3:  # Min epochs guard
            print(f"\n✓ Converged at epoch {epoch}!")
            break

    # 3. Final assembly
    print(f"\n{'='*60}")
    print(f"  Dispatching final assembly...")
    print(f"{'='*60}")

    os.chdir(CHANNELS_DIR)
    from loop_orchestrator import assemble_final_svg
    output = assemble_final_svg()
    print(f"  Output: {output}")


if __name__ == "__main__":
    from typing import Tuple
    main()
