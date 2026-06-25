#!/usr/bin/env python3
"""
start_live_loop.py — 启动 Cell Pub/Sub Loop 的管理者 + Server

这个脚本是管理者入口：
  1. 启动 server.py (FastAPI, 提供 /api/cell/publish 端点)
  2. 设置 ASTRO_LIVE_AGENTS=1 让 dispatch_cell_agent 真正走 claude.hk.cn
  3. 调用 loop_orchestrator.run_loop() 开始多 epoch 收敛循环
  4. 每个 epoch 里 run_all_cells() 会 dispatch 小弟 Claude
  5. 小弟 Claude web_search 学术特征 → 计算参数 → POST /api/cell/publish
  6. DataNotifier 广播 → SSE 推送 → GPURenderLoop 更新

用法:
  python3 start_live_loop.py                    # 默认 10 epochs
  python3 start_live_loop.py --max-epochs 20    # 自定义 epoch 数
  python3 start_live_loop.py --port 8001        # 自定义端口
  python3 start_live_loop.py --cells encoder,decoder  # 只 dispatch 指定 cells
"""
import os
import sys
import time
import json
import signal
import argparse
import threading
import urllib.request

def main():
    parser = argparse.ArgumentParser(description="Cell Pub/Sub Loop Manager")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cells", type=str, default="",
                        help="Comma-separated cell IDs to dispatch (empty=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use local hash instead of live Claude dispatch")
    args = parser.parse_args()

    # ── Set environment ──────────────────────────────────────────────────
    os.environ["ASTRO_LIVE_AGENTS"] = "0" if args.dry_run else "1"
    os.environ["ASTRO_SERVER_URL"] = f"http://127.0.0.1:{args.port}"

    if args.cells:
        os.environ["ASTRO_CELL_FILTER"] = args.cells

    project_root = os.path.dirname(os.path.abspath(__file__))
    channels_dir = os.path.join(project_root, "channels")

    print("=" * 60)
    print("  Cell Pub/Sub Loop — Live Manager")
    print(f"  ASTRO_LIVE_AGENTS = {os.environ['ASTRO_LIVE_AGENTS']}")
    print(f"  Server port       = {args.port}")
    print(f"  Max epochs        = {args.max_epochs}")
    print(f"  Cell filter       = {args.cells or '(all)'}")
    print("=" * 60)

    # ── Start server in background ───────────────────────────────────────
    import uvicorn

    def run_server():
        os.chdir(project_root)
        uvicorn.run("server:app", host="127.0.0.1", port=args.port,
                     log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    for attempt in range(10):
        time.sleep(1)
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/api/health", timeout=3)
            health = json.loads(resp.read())
            print(f"\n[Manager] Server ready: {health['status']}")
            break
        except Exception:
            if attempt == 9:
                print("[Manager] ERROR: Server failed to start after 10s")
                sys.exit(1)

    # ── Run the loop ─────────────────────────────────────────────────────
    os.chdir(channels_dir)
    sys.path.insert(0, channels_dir)
    sys.path.insert(0, project_root)  # for 'channels.rendering...' imports

    import importlib
    import loop_orchestrator
    importlib.reload(loop_orchestrator)

    print(f"\n[Manager] Starting run_loop(max_epochs={args.max_epochs})")
    print(f"[Manager] Mode: {'LIVE (claude.hk.cn dispatch)' if not args.dry_run else 'DRY RUN (local hash)'}")
    print()

    try:
        result = loop_orchestrator.run_loop(max_epochs=args.max_epochs)
        print(f"\n[Manager] Loop complete. Output length: {len(result)} chars")
    except KeyboardInterrupt:
        print("\n[Manager] Interrupted by user")
    except Exception as e:
        print(f"\n[Manager] Error: {e}")
        import traceback
        traceback.print_exc()

    # ── Push results to git ──────────────────────────────────────────────
    os.chdir(project_root)
    epoch_file = os.path.join(channels_dir, "skeleton", "epoch.json")
    if os.path.exists(epoch_file):
        with open(epoch_file) as f:
            epoch_info = json.load(f)
        epoch = epoch_info.get("current", "?")
        status = epoch_info.get("status", "?")
        print(f"\n[Manager] Final state: epoch={epoch} status={status}")

    print("\n[Manager] Done. Server still running for SSE clients.")
    print("  Press Ctrl+C to stop.")

    # Keep server alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[Manager] Shutting down.")


if __name__ == "__main__":
    main()
