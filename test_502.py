#!/usr/bin/env python3
"""
astro-svgfigure 502 诊断测试
==============================
自动加载 .env，逐步测试后端各环节，定位 502 根因。

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
# 加载 .env
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SKYNET_VENV_DIR = "/root/dylan/skynetCheapBuy/skynetCheapBuy"

# 加载项目 .env
env_file = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    print(f"✅ 加载 .env: {env_file}")
else:
    print(f"⚠️  未找到 .env: {env_file}")
    print(f"   请先: cp .env.example .env && 填入 GEMINI_API_KEY")

# 添加项目路径
sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def test_0_env_check():
    """测试 0: 环境变量检查"""
    print("\n" + "=" * 60)
    print("🧪 测试 0: 环境变量 & API Key 检查")
    print("=" * 60)

    keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "CLAUDE_COMPATIBLE_API_KEY": os.environ.get("CLAUDE_COMPATIBLE_API_KEY", ""),
    }

    has_any = False
    for name, val in keys.items():
        if val:
            masked = f"{val[:6]}...{val[-4:]}" if len(val) > 10 else "(short)"
            print(f"  ✅ {name} = {masked}")
            has_any = True
        else:
            print(f"  ❌ {name} = (empty)")

    if not has_any:
        print("\n  🚨 没有任何 API key! 后端将无法调用 LLM。")
        print("     请编辑 .env 文件，至少填入 GEMINI_API_KEY")
        return False

    print(f"\n  ✅ 至少有一个 API key 可用")
    return True


async def test_1_config_load():
    """测试 1: backend/config.py 加载"""
    print("\n" + "=" * 60)
    print("🧪 测试 1: backend/config.py Settings 加载")
    print("=" * 60)

    try:
        from backend.config import get_settings
        settings = get_settings()
        print(f"  ✅ Settings 加载成功")
        print(f"  📋 GEMINI_API_KEY: {'✅ set' if settings.GEMINI_API_KEY else '❌ empty'}")
        print(f"  📋 OPENAI_API_KEY: {'✅ set' if settings.OPENAI_API_KEY else '❌ empty'}")
        print(f"  📋 ANTHROPIC_API_KEY: {'✅ set' if settings.ANTHROPIC_API_KEY else '❌ empty'}")
        print(f"  📋 CLAUDE_COMPATIBLE_API_KEY: {'✅ set' if settings.CLAUDE_COMPATIBLE_API_KEY else '❌ empty'}")
        print(f"  📋 DEFAULT_AI_MODEL: {settings.DEFAULT_AI_MODEL}")
        print(f"  📋 CORS_ORIGINS: {settings.CORS_ORIGINS}")
        print(f"  📋 AVAILABLE_MODELS: {list(settings.AVAILABLE_MODELS.keys())}")
        return settings
    except Exception as e:
        print(f"  ❌ Settings 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_2_ai_engine_init(settings):
    """测试 2: AIEngine 初始化"""
    print("\n" + "=" * 60)
    print("🧪 测试 2: AIEngine 初始化 & Provider 路由")
    print("=" * 60)

    try:
        from backend.ai_engine import AIEngine
        engine = AIEngine(settings)
        providers = list(engine._providers.keys())
        print(f"  ✅ AIEngine 初始化成功")
        print(f"  📋 可用 Providers: {providers}")

        if not providers:
            print(f"\n  🚨 没有可用的 Provider!")
            print(f"     → 这就是 502 的根因: 后端无法调用任何 LLM")
            print(f"     → 解决: 在 .env 中填入至少一个 API key")
            return None

        return engine
    except Exception as e:
        print(f"  ❌ AIEngine 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_3_ai_completion(engine):
    """测试 3: 实际调用 LLM"""
    print("\n" + "=" * 60)
    print("🧪 测试 3: LLM 调用 (简单 completion)")
    print("=" * 60)

    try:
        model = None  # 使用默认模型
        print(f"  📡 调用默认模型...")

        result = await engine.get_completion(
            messages=[
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "Test"},
            ],
            model=model,
            temperature=0.0,
            max_tokens=10,
        )

        content = result.get("content", "")
        model_used = result.get("model", "unknown")
        print(f"  ✅ LLM 响应成功!")
        print(f"  📋 model: {model_used}")
        print(f"  📋 content: {content[:100]}")
        return True
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n  💡 可能原因:")
        print(f"     - API key 无效或过期")
        print(f"     - 网络不通 (防火墙/代理)")
        print(f"     - 模型名错误")
        return False


async def test_4_topology_gen(engine):
    """测试 4: topology_gen 端到端"""
    print("\n" + "=" * 60)
    print("🧪 测试 4: Pipeline topology_gen 端到端")
    print("=" * 60)

    try:
        from backend.pipeline.topology_gen import generate_topology
        from backend.schemas import ElkAlgorithm, ElkDirection

        text = "A simple neural network with input layer, hidden layer, and output layer connected sequentially."
        print(f"  📝 Input: {text[:60]}...")
        print(f"  📡 生成拓扑中 (可能需要 5-15 秒)...")

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
            print(f"  ✅ 拓扑生成成功!")
            print(f"  📋 model: {result.model_used}")
            print(f"  📋 nodes: {nodes}, edges: {edges}")
            if topo:
                print(f"  📋 node ids: {[c.id for c in topo.children]}")
            return True
        else:
            print(f"  ❌ 拓扑生成失败: {result.error}")
            return False
    except Exception as e:
        print(f"  ❌ topology_gen 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_http_endpoint():
    """测试 5: HTTP 端点直连 (模拟 Astro 的 fetch)"""
    print("\n" + "=" * 60)
    print("🧪 测试 5: HTTP 端点 curl 模拟 (localhost:8000)")
    print("=" * 60)

    try:
        import httpx
    except ImportError:
        print("  ⚠️  httpx 未安装，改用 urllib")
        import urllib.request
        import urllib.error

        url = "http://localhost:8000/api/models"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                print(f"  ✅ GET /api/models → {list(data.keys())}")
        except urllib.error.URLError as e:
            print(f"  ❌ 连接 localhost:8000 失败: {e}")
            print(f"     → 后端没有运行! 请在另一个终端: python server.py")
            return False
        except Exception as e:
            print(f"  ❌ 请求失败: {e}")
            return False
        return True

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test /api/models
        try:
            resp = await client.get("http://localhost:8000/api/models")
            print(f"  ✅ GET /api/models → {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"     providers: {list(data.keys())}")
            else:
                print(f"     body: {resp.text[:200]}")
        except httpx.ConnectError:
            print(f"  ❌ 连接 localhost:8000 失败!")
            print(f"     → 后端没有运行! 请在另一个终端: python server.py")
            return False

        # Test /api/topology
        try:
            resp = await client.post(
                "http://localhost:8000/api/topology",
                json={"text": "encoder decoder architecture", "model": None},
                timeout=60.0,
            )
            print(f"  {'✅' if resp.status_code == 200 else '❌'} POST /api/topology → {resp.status_code}")
            body = resp.json()
            if resp.status_code == 200:
                print(f"     success: {body.get('success')}")
                topo = body.get("topology", {})
                print(f"     nodes: {len(topo.get('children', []))}")
            else:
                print(f"     error: {body.get('error', body)}")
        except Exception as e:
            print(f"  ❌ POST /api/topology 失败: {e}")

    return True


async def main():
    print("=" * 60)
    print("🔧 astro-svgfigure 502 诊断测试")
    print(f"📁 项目: {PROJECT_DIR}")
    print("=" * 60)

    # Test 0: env check
    has_keys = await test_0_env_check()
    if not has_keys:
        print("\n🛑 没有 API key，无法继续。请先配置 .env")
        return

    # Test 1: config
    settings = await test_1_config_load()
    if not settings:
        return

    # Test 2: AI engine
    engine = await test_2_ai_engine_init(settings)
    if not engine:
        return

    # Test 3: LLM call
    llm_ok = await test_3_ai_completion(engine)
    if not llm_ok:
        print("\n🛑 LLM 调用失败，topology_gen 也会失败")
        return

    # Test 4: topology gen
    topo_ok = await test_4_topology_gen(engine)

    # Test 5: HTTP endpoint (only if backend is running separately)
    print("\n  💡 测试 5 需要后端在另一个终端运行 (python server.py)")
    print("     如果后端没运行会报连接错误，这是正常的。")
    await test_5_http_endpoint()

    # Summary
    print("\n" + "=" * 60)
    print("📊 诊断结论")
    print("=" * 60)

    if topo_ok:
        print("  ✅ 后端 Pipeline 完全正常!")
        print("  → 502 原因很可能是:")
        print("    1. 前端启动时后端还没 ready")
        print("    2. 先启动 python server.py，再启动 bun dev")
        print("    3. 或用: bun run dev:all (同时启动)")
    else:
        print("  ❌ Pipeline 有问题，检查上面的错误信息")

    print("\n🚀 推荐操作:")
    print("  终端 1: python server.py     # 先启动后端")
    print("  终端 2: bun dev              # 再启动前端")
    print("  浏览器: http://localhost:4321/generate")


if __name__ == "__main__":
    asyncio.run(main())
