#!/usr/bin/env python3
"""
管理者Claude的小弟调度器 — 串行派发，每个小弟完成一个task后再派下一个。
通过Anthropic API调用claude-sonnet-4-6作为小弟。
"""
import json, os, sys, time, urllib.request, urllib.error

API_URL = "https://api.anthropic.com/v1/messages"

def dispatch_xiaodi(xiaodi_id, task_prompt, system_prompt="You are xiaodi (小弟), a sub-Claude worker. Complete the task and return ONLY the result."):
    """派一个小弟出去干活"""
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": task_prompt}]
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "placeholder",  # handled by cookie auth
        "anthropic-version": "2023-06-01"
    }
    
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST"
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
            elapsed = time.time() - start
            print(f"[xiaodi #{xiaodi_id}] OK ({elapsed:.1f}s) — {len(text)} chars")
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"[xiaodi #{xiaodi_id}] HTTP {e.code}: {body[:200]}")
        return None
    except Exception as e:
        print(f"[xiaodi #{xiaodi_id}] ERROR: {e}")
        return None


if __name__ == "__main__":
    # Test: dispatch 1 xiaodi to verify API works
    print("=== Testing single xiaodi dispatch ===")
    result = dispatch_xiaodi(
        111,
        "Count from 1 to 5 and say 'xiaodi #111 reporting for duty'. Nothing else."
    )
    if result:
        print(f"Response: {result[:200]}")
    else:
        print("FAILED — API not available, need cookie auth or API key")
