#!/usr/bin/env python3
"""
astro-svgfigure 502 诊断测试 v2
=================================
自动加载 .env (项目 + CheapBuy)，逐步测试后端各环节。

使用方式:
    cd /root/dylan/skynetCheapBuy/astro-svgfigure
    python3 test_502.py
"""

import os
import sys
import json
import asyncio
import logging

# ============================================================================
# 加载 .env (项目优先, CheapBuy fallback)
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHEAPBUY_DIR = "/root/dylan/CheapBuy"

def load_env(path):
    loaded = 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                        loaded += 1
        print(f"  ✅ 加载 {path} ({loaded} vars)")
    return loaded

print("📂 加载环境变量:")
load_env(os.path.join(PROJECT_DIR, ".env"))
load_env(os.path.join(CHEAPBUY_DIR, ".env"))

sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_0_env_check():
    """测试 0: 环境变量"""
    print("\n" + "=" * 60)
    print("🧪 测试 0: API Key 检查")
    print("=" * 60)

    keys_to_check = [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "ANTHROPIC_API_KEY",
        "CLAUDE_COMPATIBLE_API_KEY",
        "CLAUDE_COMPATIBLE_API_BASE",
    ]

    has_any_key = False
    for name in keys_to_check:
        val = os.environ.get(name, "")
        if val:
            if "KEY" in name:
                masked = f"{val[:6]}...{val[-4:]}" if len(val) > 10 else "(short)"
                has_any_key = True
            else:
                masked = val
            print(f"  ✅ {name} = {masked}")
        else:
            print(f"  ⬚  {name} = (empty)")

    if not has_any_key:
        print("\n  🚨 没有任何 API key!")
        return False

    return True


async def test_1_config():
    """测试 1: Settings"""
    print("\n" + "=" * 60)
    print("🧪 测试 1: backend/config.py")
    print("=" * 60)

    try:
        from backend.config import get_settings
        get_settings.cache_clear()
        settings = get_settings()
        print(f"  ✅ Settings OK")
        print(f"  GEMINI_API_KEY: {'✅' if settings.GEMINI_API_KEY else '❌'}")
        print(f"  OPENAI_API_KEY: {'✅' if settings.OPENAI_API_KEY else '❌'} base={settings.OPENAI_API_BASE}")
        print(f"  ANTHROPIC_API_KEY: {'✅' if settings.ANTHROPIC_API_KEY else '❌'}")
        print(f"  CLAUDE_COMPATIBLE: {'✅' if settings.CLAUDE_COMPATIBLE_API_KEY else '❌'} base={settings.CLAUDE_COMPATIBLE_API_BASE}")
        print(f"  DEFAULT_AI_MODEL: {settings.DEFAULT_AI_MODEL}")
        models = settings.AVAILABLE_MODELS
        print(f"  AVAILABLE_MODELS: {list(models.keys())} ({sum(len(v) for v in models.values())} total)")
        return settings
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback; traceback.print_exc()
        return None


async def test_2_ai_engine(settings):
    """测试 2: AIEngine"""
    print("\n" + "=" * 60)
    print("🧪 测试 2: AIEngine 初始化")
    print("=" * 60)

    try:
        from backend.ai_engine import AIEngine
        engine = AIEngine(settings)
        providers = list(engine._providers.keys())
        print(f"  ✅ Providers: {providers}")

        if not providers:
            print(f"  🚨 没有可用 Provider! 这就是 502 根因。")
            return None

        return engine
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback; traceback.print_exc()
        return None


async def test_3_llm_call(engine):
    """测试 3: LLM 调用"""
    print("\n" + "=" * 60)
    print("🧪 测试 3: LLM 简单调用")
    print("=" * 60)

    try:
        result = await engine.get_completion(
            messages=[
                {"role": "system", "content": "Reply with exactly one word: OK"},
                {"role": "user", "content": "Test"},
            ],
            model=None,
            temperature=0.0,
            max_tokens=10,
        )
        print(f"  ✅ model={result.get('model','?')}")
        print(f"  ✅ content={result.get('content','')[:100]}")
        return True
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_4_topology(engine):
    """测试 4: topology_gen"""
    print("\n" + "=" * 60)
    print("🧪 测试 4: topology_gen 端到端")
    print("=" * 60)

    try:
        from backend.pipeline.topology_gen import generate_topology
        from backend.schemas import ElkAlgorithm, ElkDirection

        text = "A simple encoder-decoder with attention mechanism"
        print(f"  📝 Input: {text}")
        print(f"  ⏳ 生成中 (5-15s)...")

        result = await generate_topology(
            ai_engine=engine,
            text=text,
            model=None,
            algorithm=ElkAlgorithm.LAYERED,
            direction=ElkDirection.DOWN,
        )

        if result.success:
            topo = result.topology
            nodes = len(topo.children) if topo else 0
            edges = len(topo.edges) if topo else 0
            print(f"  ✅ 成功! model={result.model_used}, nodes={nodes}, edges={edges}")
            if topo:
                print(f"  📋 nodes: {[c.id for c in topo.children]}")
            return True
        else:
            print(f"  ❌ 失败: {result.error}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_5_http():
    """测试 5: HTTP 端点"""
    print("\n" + "=" * 60)
    print("🧪 测试 5: HTTP localhost:8000 连通性")
    print("=" * 60)

    import urllib.request
    import urllib.error

    # GET /api/models
    try:
        req = urllib.request.Request("http://localhost:8000/api/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"  ✅ GET /api/models → {list(data.keys())}")
    except urllib.error.URLError as e:
        print(f"  ❌ 连接失败: {e}")
        print(f"     → 后端没运行! 在另一终端: python server.py")
        return False

    # POST /api/topology
    try:
        payload = json.dumps({"text": "encoder decoder", "model": None}).encode()
        req = urllib.request.Request(
            "http://localhost:8000/api/topology",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            print(f"  ✅ POST /api/topology → success={data.get('success')}")
            if data.get("topology"):
                t = data["topology"]
                print(f"     nodes={len(t.get('children',[]))}, edges={len(t.get('edges',[]))}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ❌ POST /api/topology → {e.code}")
        print(f"     body: {body[:500]}")
    except Exception as e:
        print(f"  ❌ POST /api/topology 异常: {e}")

    return True


async def test_6_tryallai_direct():
    """测试 6: 直接测试 tryallai.com"""
    print("\n" + "=" * 60)
    print("🧪 测试 6: tryallai.com 直连")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CLAUDE_COMPATIBLE_API_KEY") or ""
    api_base = os.environ.get("OPENAI_API_BASE", "")
    compat_base = os.environ.get("CLAUDE_COMPATIBLE_API_BASE", "")

    if not api_key:
        print("  ⬚  跳过 (无 API key)")
        return True

    # 试 OpenAI 格式
    if api_base:
        import urllib.request
        url = f"{api_base.rstrip('/')}/chat/completions"
        payload = json.dumps({
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }).encode()

        print(f"  📡 测试 OpenAI 格式: {url}")
        try:
            req = urllib.request.Request(url, data=payload, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  ✅ OpenAI格式成功: {content[:80]}")
        except Exception as e:
            print(f"  ❌ OpenAI格式失败: {e}")

    # 试 Claude 格式
    if compat_base:
        import urllib.request
        url = f"{compat_base.rstrip('/')}/v1/messages"
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }).encode()

        print(f"  📡 测试 Claude 格式: {url}")
        try:
            req = urllib.request.Request(url, data=payload, headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                print(f"  ✅ Claude格式成功: {content[:80]}")
        except Exception as e:
            print(f"  ❌ Claude格式失败: {e}")

    return True


async def main():
    print("=" * 60)
    print("🔧 astro-svgfigure 502 诊断 v2")
    print(f"📁 {PROJECT_DIR}")
    print("=" * 60)

    has_keys = await test_0_env_check()
    if not has_keys:
        print("\n🛑 无 API key")
        return

    settings = await test_1_config()
    if not settings:
        return

    engine = await test_2_ai_engine(settings)
    if not engine:
        return

    llm_ok = await test_3_llm_call(engine)

    if llm_ok:
        await test_4_topology(engine)
    else:
        print("\n  ⏭️  跳过 test_4")

    print("\n  💡 测试 5 需后端在另一终端运行")
    await test_5_http()

    await test_6_tryallai_direct()

    print("\n" + "=" * 60)
    print("📊 结论")
    print("=" * 60)
    if llm_ok:
        print("  ✅ Pipeline 正常! 502 = 启动顺序问题")
        print("  → 先: python server.py  再: bun dev")
    else:
        print("  ❌ 检查上面的错误信息")


if __name__ == "__main__":
    asyncio.run(main())
