"""
AI Engine — Multi-Provider Abstraction Factory
================================================
Fully modeled after skynetCheapBuy/app/core/ai_engine.py:
  AIProvider (abstract base)
    ├── OpenAIProvider          — gpt-*, o1-*, o3-* via openai AsyncOpenAI
    ├── AnthropicProvider       — claude-* via anthropic AsyncAnthropic (native)
    ├── GoogleProvider          — gemini-* via google-generativeai
    └── ClaudeCompatibleProvider — claude-* via /v1/messages (third-party, httpx)
  AIEngine
    └── _get_provider(model) → auto-routes to the correct provider

Client reference: 请求第三方openai_anthropic流程.py
  - Tests tool calling with ClaudeCompatibleProvider
  - Validates content_blocks, tool_uses, stop_reason fields
  - Backward-compatible: result["content"] is always str

GitHub references:
  - dylanyunlon/skynetCheapBuy/app/core/ai_engine.py
  - openai/openai-python
  - anthropics/anthropic-sdk-python
  - google/generative-ai-python
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: model name detection (matches skynetCheapBuy logic)
# ============================================================================

def is_openai_model(model: str) -> bool:
    """Check if model is an OpenAI model (gpt-*, o1-*, o3-*)."""
    m = model.lower()
    return m.startswith(("gpt-", "o1-", "o3-"))


def is_claude_model(model: str) -> bool:
    """Check if model is a Claude model."""
    m = model.lower()
    return m.startswith(("claude-", "claude_"))


def is_gemini_model(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.lower().startswith("gemini")


# ============================================================================
# Abstract Base: AIProvider
# ============================================================================

class AIProvider(ABC):
    """
    Abstract AI Provider base class.
    Each provider implements:
      - get_completion(): single-shot request → dict
      - stream_completion(): streaming request → async generator
    """

    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Non-streaming completion.

        Returns dict with at least:
          - content: str (text response, backward compat)
          - content_blocks: list (raw content blocks)
          - tool_uses: list (tool call blocks, may be empty)
          - stop_reason: str
          - model: str
          - usage: dict
        """
        ...

    @abstractmethod
    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming completion.
        Yields dicts with:
          - type: "text_delta" | "tool_use" | "done" | "error"
          - content / tool / ...
        """
        ...


# ============================================================================
# Provider: OpenAI (gpt-*, o1-*, o3-*)
# ============================================================================

class OpenAIProvider(AIProvider):
    """
    OpenAI provider using openai.AsyncOpenAI.
    Also works with OpenAI-compatible endpoints (e.g., OpenRouter, Bianxie).

    Reference: openai/openai-python
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAIProvider initialized (base_url={base_url})")

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Convert tool format: our schema uses name/description/input_schema
        # OpenAI expects type:"function" wrapping
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                })
            params["tools"] = openai_tools

        response = await self._client.chat.completions.create(**params)
        choice = response.choices[0]
        message = choice.message

        # Build unified response
        content_blocks = []
        tool_uses = []

        if message.content:
            content_blocks.append({"type": "text", "text": message.content})

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_block = {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments) if tc.function.arguments else {},
                }
                content_blocks.append(tool_block)
                tool_uses.append(tool_block)

        return {
            "content": message.content or "",
            "content_blocks": content_blocks,
            "tool_uses": tool_uses,
            "tool_calls": tool_uses if tool_uses else None,  # backward compat
            "stop_reason": choice.finish_reason or "stop",
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                })
            params["tools"] = openai_tools

        stream = await self._client.chat.completions.create(**params)

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                yield {"type": "text_delta", "content": delta.content}

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield {
                        "type": "tool_use",
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function and tc.function.name else "",
                        "arguments_delta": tc.function.arguments if tc.function else "",
                    }

            if chunk.choices[0].finish_reason:
                yield {"type": "done", "stop_reason": chunk.choices[0].finish_reason}


# ============================================================================
# Provider: Anthropic (native SDK)
# ============================================================================

class AnthropicProvider(AIProvider):
    """
    Native Anthropic Claude provider using anthropic.AsyncAnthropic.

    Reference: anthropics/anthropic-sdk-python
    """

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        logger.info(f"AnthropicProvider initialized (base_url={base_url})")

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        # Extract system message if present (Anthropic uses separate system param)
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        params: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_msg:
            params["system"] = system_msg

        if tools:
            params["tools"] = tools  # Anthropic native format matches ours

        response = await self._client.messages.create(**params)

        # Build unified response
        content_blocks = []
        tool_uses = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_block = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                content_blocks.append(tool_block)
                tool_uses.append(tool_block)

        return {
            "content": "\n".join(text_parts),
            "content_blocks": content_blocks,
            "tool_uses": tool_uses,
            "tool_calls": tool_uses if tool_uses else None,
            "stop_reason": response.stop_reason or "end_turn",
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        }

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        params: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_msg:
            params["system"] = system_msg
        if tools:
            params["tools"] = tools

        async with self._client.messages.stream(**params) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield {"type": "text_delta", "content": event.delta.text}
                        elif hasattr(event.delta, "partial_json"):
                            yield {"type": "tool_input_delta", "content": event.delta.partial_json}
                    elif event.type == "content_block_start":
                        if hasattr(event.content_block, "type") and event.content_block.type == "tool_use":
                            yield {
                                "type": "tool_use_start",
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                            }
                    elif event.type == "message_stop":
                        yield {"type": "done", "stop_reason": "end_turn"}


# ============================================================================
# Provider: Google Gemini
# ============================================================================

class GoogleProvider(AIProvider):
    """
    Google Gemini provider using google-generativeai SDK.

    Reference: google/generative-ai-python
    """

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai required: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self._genai = genai
        logger.info("GoogleProvider (Gemini) initialized")

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        gen_model = self._genai.GenerativeModel(model)

        # Convert messages to Gemini format
        gemini_contents = self._convert_messages(messages)

        generation_config = self._genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Gemini tool calling
        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools(tools)

        response = await gen_model.generate_content_async(
            gemini_contents,
            generation_config=generation_config,
            tools=gemini_tools,
        )

        # Build unified response
        content_blocks = []
        tool_uses = []
        text_parts = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content_blocks.append({"type": "text", "text": part.text})
                    text_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_block = {
                        "type": "tool_use",
                        "id": f"tool_{fc.name}",
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    }
                    content_blocks.append(tool_block)
                    tool_uses.append(tool_block)

        # Usage
        usage_meta = getattr(response, "usage_metadata", None)

        return {
            "content": "\n".join(text_parts),
            "content_blocks": content_blocks,
            "tool_uses": tool_uses,
            "tool_calls": tool_uses if tool_uses else None,
            "stop_reason": "stop",
            "model": model,
            "usage": {
                "input_tokens": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
                "output_tokens": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
            },
        }

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        gen_model = self._genai.GenerativeModel(model)
        gemini_contents = self._convert_messages(messages)

        generation_config = self._genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools(tools)

        response = await gen_model.generate_content_async(
            gemini_contents,
            generation_config=generation_config,
            tools=gemini_tools,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield {"type": "text_delta", "content": chunk.text}

        yield {"type": "done", "stop_reason": "stop"}

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict]:
        """Convert OpenAI-style messages to Gemini format."""
        contents = []
        system_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = "user" if role == "user" else "model"

            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                        elif item.get("type") == "image_url":
                            # Handle base64 images
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                import base64
                                # Extract mime and data
                                header, b64data = url.split(",", 1)
                                mime = header.split(";")[0].split(":")[1]
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime,
                                        "data": b64data,
                                    }
                                })
            else:
                parts = [{"text": str(content)}]

            # Prepend system message to first user message
            if system_parts and gemini_role == "user":
                parts = [{"text": "\n".join(system_parts) + "\n\n"}] + parts
                system_parts = []

            contents.append({"role": gemini_role, "parts": parts})

        return contents

    def _convert_tools(self, tools: List[Dict]) -> List:
        """Convert our tool format to Gemini function declarations."""
        declarations = []
        for t in tools:
            declarations.append(
                self._genai.protos.Tool(
                    function_declarations=[
                        self._genai.protos.FunctionDeclaration(
                            name=t["name"],
                            description=t.get("description", ""),
                            parameters=self._schema_to_proto(
                                t.get("input_schema", t.get("parameters", {}))
                            ),
                        )
                    ]
                )
            )
        return declarations

    def _schema_to_proto(self, schema: Dict) -> Any:
        """Convert JSON Schema to Gemini proto format (simplified)."""
        # For complex schemas, use genai.protos.Schema
        # Simplified version that handles common cases
        return schema


# ============================================================================
# Provider: Claude-Compatible (third-party /v1/messages endpoint)
# ============================================================================

class ClaudeCompatibleProvider(AIProvider):
    """
    Claude-compatible provider via /v1/messages endpoint using httpx.
    Used for third-party Claude API providers (e.g., Bianxie, OpenRouter).

    This matches the test flow in 请求第三方openai_anthropic流程.py:
      - Sends to /v1/messages with anthropic-version header
      - Returns content_blocks, tool_uses, stop_reason
      - Backward compat: content is str

    Reference: 请求第三方openai_anthropic流程.py → test_2_claude_provider_tools
    """

    def __init__(self, api_key: str, base_url: str):
        if not base_url:
            raise ValueError("ClaudeCompatibleProvider requires base_url")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"ClaudeCompatibleProvider initialized (base_url={self._base_url})")

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        # Separate system message
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_msg:
            payload["system"] = system_msg

        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self._base_url}/v1/messages"
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Parse response (Anthropic /v1/messages format)
        content_blocks = []
        tool_uses = []
        text_parts = []

        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                content_blocks.append({"type": "text", "text": block["text"]})
                text_parts.append(block["text"])
            elif block_type == "tool_use":
                tool_block = {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
                content_blocks.append(tool_block)
                tool_uses.append(tool_block)

        return {
            "content": "\n".join(text_parts),
            "content_blocks": content_blocks,
            "tool_uses": tool_uses,
            "tool_calls": tool_uses if tool_uses else None,
            "stop_reason": data.get("stop_reason", "end_turn"),
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
        }

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if system_msg:
            payload["system"] = system_msg
        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self._base_url}/v1/messages"

        async with self._client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            buffer = ""

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield {"type": "done", "stop_reason": "end_turn"}
                    return

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield {"type": "text_delta", "content": delta.get("text", "")}
                    elif delta.get("type") == "input_json_delta":
                        yield {"type": "tool_input_delta", "content": delta.get("partial_json", "")}

                elif event_type == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        yield {
                            "type": "tool_use_start",
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                        }

                elif event_type == "message_stop":
                    yield {"type": "done", "stop_reason": "end_turn"}

    async def close(self):
        """Close the underlying httpx client."""
        await self._client.aclose()


# ============================================================================
# AIEngine: Factory + Unified Interface
# ============================================================================

class AIEngine:
    """
    AI Engine — auto-routes requests to the appropriate provider based on model.

    Usage (matches skynetCheapBuy pattern):
        ai_engine = AIEngine()
        result = await ai_engine.get_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gemini-2.5-flash",
        )
        print(result["content"])

    For tool calling (matches 请求第三方openai_anthropic流程.py test_2):
        tools = [{"name": "get_weather", "description": "...", "input_schema": {...}}]
        result = await ai_engine.get_completion(messages=..., model=..., tools=tools)
        assert "content_blocks" in result
        assert "tool_uses" in result
        assert "stop_reason" in result
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._providers: Dict[str, AIProvider] = {}
        self._init_providers()

    def _init_providers(self):
        """Initialize available providers based on config."""
        s = self._settings

        if s.OPENAI_API_KEY:
            try:
                self._providers["openai"] = OpenAIProvider(s.OPENAI_API_KEY, s.OPENAI_API_BASE)
            except ImportError as e:
                logger.warning(f"OpenAI provider unavailable: {e}")

        if s.ANTHROPIC_API_KEY:
            try:
                self._providers["anthropic"] = AnthropicProvider(s.ANTHROPIC_API_KEY, s.ANTHROPIC_API_BASE)
            except ImportError as e:
                logger.warning(f"Anthropic provider unavailable: {e}")

        if s.GEMINI_API_KEY:
            if s.GEMINI_API_BASE:
                # Proxy mode (e.g. tryallai): use OpenAI-compatible format for Gemini
                try:
                    self._providers["gemini"] = OpenAIProvider(s.GEMINI_API_KEY, s.GEMINI_API_BASE)
                    logger.info(f"Gemini via OpenAI-compatible proxy: {s.GEMINI_API_BASE}")
                except ImportError as e:
                    logger.warning(f"OpenAI provider unavailable for Gemini proxy: {e}")
            else:
                # Direct mode: use google-generativeai SDK
                try:
                    self._providers["gemini"] = GoogleProvider(s.GEMINI_API_KEY)
                except ImportError as e:
                    logger.warning(f"Gemini provider unavailable: {e}")

        if s.CLAUDE_COMPATIBLE_API_KEY and s.CLAUDE_COMPATIBLE_API_BASE:
            self._providers["claude_compatible"] = ClaudeCompatibleProvider(
                s.CLAUDE_COMPATIBLE_API_KEY, s.CLAUDE_COMPATIBLE_API_BASE
            )

        logger.info(f"AIEngine initialized with providers: {list(self._providers.keys())}")

    def _get_provider(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> AIProvider:
        """
        Route to the correct provider based on model name.
        Matches skynetCheapBuy/_get_provider logic:
          1. If explicit api_key provided → create ad-hoc provider
          2. Otherwise → use pre-initialized provider from settings
        """
        # Ad-hoc provider (like skynetCheapBuy when called with explicit keys)
        if api_key:
            if is_claude_model(model):
                if api_url:
                    return ClaudeCompatibleProvider(api_key, api_url)
                return AnthropicProvider(api_key)
            elif is_openai_model(model):
                return OpenAIProvider(api_key, api_url or "https://api.openai.com/v1")
            elif is_gemini_model(model):
                if api_url:
                    return OpenAIProvider(api_key, api_url)  # Proxy mode
                return GoogleProvider(api_key)  # Direct mode
            else:
                return OpenAIProvider(api_key, api_url or "https://api.openai.com/v1")

        # Pre-initialized provider routing
        if is_gemini_model(model) and "gemini" in self._providers:
            return self._providers["gemini"]
        if is_claude_model(model):
            if "anthropic" in self._providers:
                return self._providers["anthropic"]
            if "claude_compatible" in self._providers:
                return self._providers["claude_compatible"]
        if is_openai_model(model) and "openai" in self._providers:
            return self._providers["openai"]

        # Fallback: try any available provider
        if self._providers:
            first_key = next(iter(self._providers))
            logger.warning(f"No specific provider for model '{model}', falling back to '{first_key}'")
            return self._providers[first_key]

        raise RuntimeError(
            f"No provider available for model '{model}'. "
            f"Check .env: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
        )

    # ── Public API ──────────────────────────────────────────────────────

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Non-streaming completion with auto provider routing.

        Returns dict with:
          - content: str
          - content_blocks: list
          - tool_uses: list
          - stop_reason: str
          - model: str
          - usage: dict
        """
        model = model or self.DEFAULT_MODEL
        provider = self._get_provider(model, api_key=api_key, api_url=api_url)

        logger.info(f"get_completion: model={model}, provider={type(provider).__name__}")

        return await provider.get_completion(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        *,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming completion with auto provider routing."""
        model = model or self.DEFAULT_MODEL
        provider = self._get_provider(model, api_key=api_key, api_url=api_url)

        logger.info(f"stream_completion: model={model}, provider={type(provider).__name__}")

        async for chunk in provider.stream_completion(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    # ── Convenience methods ─────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Simple text generation — returns just the text content."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = await self.get_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return result["content"]

    async def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        if system:
            system += "\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown fences, no preamble."
        else:
            system = "Respond ONLY with valid JSON, no markdown fences, no preamble."

        text = await self.generate(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        return json.loads(cleaned)

    @property
    def available_providers(self) -> List[str]:
        """List of initialized provider names."""
        return list(self._providers.keys())

    @property
    def available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """Models grouped by provider."""
        return self._settings.AVAILABLE_MODELS
