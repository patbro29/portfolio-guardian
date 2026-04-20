"""
providers.py — LLM provider abstraction.

Hides all vendor-specific API details behind a single interface. agent.py calls
run_llm_agent() and receives a stream of typed events (TextEvent, ToolCallEvent,
etc.) without knowing which LLM is underneath.

To switch providers, change the PROVIDER constant or set the LLM_PROVIDER env var.
No other file needs to change.

Supported providers:
- "gemini"    → Google Gemini 2.5 Flash (free tier)
- "anthropic" → Claude Sonnet 4.6

Tool schema note:
tools.py defines schemas in Anthropic's format (input_schema key). Gemini expects
the same JSON Schema but under a parameters key. _gemini_convert_schema() handles
this translation, so tools only need to be written once.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

# ============================================================================
# PROVIDER CHOICE — change this one constant to swap LLMs
# ============================================================================
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" | "anthropic"


# ============================================================================
# Unified event format — what agent.py sees regardless of provider
# ============================================================================
# Using a dataclass makes the "contract" between agent.py and providers.py
# explicit. Any provider we add in the future must yield these event types.

@dataclass
class TextEvent:
    content: str

@dataclass
class ToolCallEvent:
    name: str
    input: dict[str, Any]
    tool_use_id: str  # opaque ID we'll need to pair with the result later

@dataclass
class ToolResultEvent:
    name: str
    result: str
    tool_use_id: str

@dataclass
class StopEvent:
    reason: str  # "end_turn" | "max_tokens" | "tool_use" | "error"


# ============================================================================
# GEMINI IMPLEMENTATION
# ============================================================================

def _gemini_convert_schema(anthropic_schema: dict) -> dict:
    """
    Convert an Anthropic-style tool schema to a Gemini FunctionDeclaration dict.

    Anthropic format:
        {"name": ..., "description": ..., "input_schema": {...JSON Schema...}}

    Gemini format:
        {"name": ..., "description": ..., "parameters": {...JSON Schema...}}

    The JSON Schema inside is essentially identical for our purposes.
    We just rename the wrapper key.
    """
    return {
        "name": anthropic_schema["name"],
        "description": anthropic_schema["description"],
        "parameters": anthropic_schema["input_schema"],
    }


def _gemini_build_tools(tool_schemas: list[dict]) -> list:
    """
    Build the Gemini `tools` argument from Anthropic-style schemas.

    Skips any web_search schema entries — Gemini's API does not allow combining
    its built-in google_search grounding with function_declarations in the same
    request. Web search is available on the Anthropic provider path only.
    """
    from google.genai import types

    function_decls = []
    for schema in tool_schemas:
        # Skip web_search — it uses Anthropic's server-tool format and has no
        # Gemini equivalent that can be combined with function_declarations.
        if schema.get("type", "").startswith("web_search"):
            continue
        function_decls.append(_gemini_convert_schema(schema))

    tools = []
    if function_decls:
        tools.append(types.Tool(function_declarations=function_decls))

    return tools


def _gemini_run_agent(
    user_message: str,
    holdings: list[dict],
    system_prompt: str,
    tool_schemas: list[dict],
    tool_executor,  # callable: (name, input) -> str
    max_turns: int,
) -> Iterator[Any]:
    """
    Run the agent loop against the Gemini API.

    Uses a chat session (history managed automatically) with automatic function
    calling disabled — we drive the loop manually so we can yield events at each
    step for the UI. Each turn: extract text and function_call parts, execute any
    tool calls, send results back as function_response parts, repeat until the
    model produces a final answer with no function calls.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or _secret("GEMINI_API_KEY")
    if not api_key:
        yield StopEvent(reason="error")
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    tools = _gemini_build_tools(tool_schemas)

    # Seed the conversation with holdings + the user's question.
    initial_msg = (
        f"Here are my current holdings:\n"
        f"{json.dumps(holdings, indent=2)}\n\n"
        f"My question: {user_message}"
    )

    # Build the config. Disable auto function calling so WE control the loop.
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Gemini's chat API keeps history for us. Simpler than managing messages[] manually.
    chat = client.chats.create(model="gemini-2.5-flash", config=config)

    # First send: user message with holdings.
    response = chat.send_message(initial_msg)

    for turn in range(max_turns):
        # Extract any text the model produced.
        text_parts = []
        function_calls = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)

        if text_parts:
            yield TextEvent(content="".join(text_parts))

        # If no function calls, the model is done.
        if not function_calls:
            yield StopEvent(reason="end_turn")
            return

        # Execute each function call, collect results.
        result_parts = []
        for fc in function_calls:
            # Gemini function calls have .name and .args (already a dict)
            fc_args = dict(fc.args) if fc.args else {}

            yield ToolCallEvent(
                name=fc.name,
                input=fc_args,
                tool_use_id=fc.name,  # Gemini doesn't expose IDs; we use name
            )

            result_str = tool_executor(fc.name, fc_args)

            yield ToolResultEvent(
                name=fc.name,
                result=result_str,
                tool_use_id=fc.name,
            )

            # Gemini wants the result wrapped in function_response.
            # The `response` must be a dict; we parse our JSON string back.
            try:
                result_obj = json.loads(result_str)
            except json.JSONDecodeError:
                result_obj = {"result": result_str}

            result_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response=result_obj if isinstance(result_obj, dict) else {"result": result_obj},
                )
            )

        # Send results back to the model — it will either finalize or call more tools.
        response = chat.send_message(result_parts)

    yield StopEvent(reason="max_tokens")


# ============================================================================
# ANTHROPIC IMPLEMENTATION (stub — preserved for easy future swap)
# ============================================================================

def _anthropic_run_agent(
    user_message: str,
    holdings: list[dict],
    system_prompt: str,
    tool_schemas: list[dict],
    tool_executor,
    max_turns: int,
) -> Iterator[Any]:
    """
    Claude Sonnet 4.6 via Anthropic SDK. Same loop structure as the Gemini
    implementation — kept here as an alternative provider. Switch by setting
    PROVIDER = 'anthropic' or LLM_PROVIDER=anthropic in the environment.
    """
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API_KEY")
    if not api_key:
        yield StopEvent(reason="error")
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    initial_content = (
        f"Here are my current holdings:\n"
        f"{json.dumps(holdings, indent=2)}\n\n"
        f"My question: {user_message}"
    )
    messages = [{"role": "user", "content": initial_content}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            tools=tool_schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                yield TextEvent(content=block.text)

        if response.stop_reason != "tool_use":
            yield StopEvent(reason=response.stop_reason)
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "web_search":
                yield ToolCallEvent(name="web_search", input=block.input, tool_use_id=block.id)
                continue

            yield ToolCallEvent(name=block.name, input=block.input, tool_use_id=block.id)
            result_str = tool_executor(block.name, block.input)
            yield ToolResultEvent(name=block.name, result=result_str, tool_use_id=block.id)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    yield StopEvent(reason="max_tokens")


# ============================================================================
# PUBLIC API — what agent.py calls
# ============================================================================

def run_llm_agent(
    user_message: str,
    holdings: list[dict],
    system_prompt: str,
    tool_schemas: list[dict],
    tool_executor,
    max_turns: int = 10,
) -> Iterator[Any]:
    """
    Dispatch to the configured provider. Returns an iterator of events.

    agent.py calls this. agent.py does not know which provider is active.
    """
    if PROVIDER == "gemini":
        yield from _gemini_run_agent(
            user_message, holdings, system_prompt, tool_schemas, tool_executor, max_turns
        )
    elif PROVIDER == "anthropic":
        yield from _anthropic_run_agent(
            user_message, holdings, system_prompt, tool_schemas, tool_executor, max_turns
        )
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


def _secret(key: str) -> str | None:
    """Try to read from Streamlit secrets (for deployed app)."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None
