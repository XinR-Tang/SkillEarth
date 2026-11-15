#!/usr/bin/env python3
"""Run one Earth-Bench task with SkillEarth retrieval and live MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


SYSTEM_PROMPT = (
    "You are a geospatial analysis agent. Solve the user's Earth-observation task using the "
    "available MCP tools. Ground every claim in tool observations. When annotations or other "
    "artifacts are requested, create them with the relevant tool and include their returned paths "
    "in the final answer. Give a concise human-readable final answer, not a multiple-choice format."
)


def features(text: str) -> list[str]:
    normalized = " ".join(str(text or "").lower().split())
    words = re.findall(r"[a-z0-9_]+", normalized)
    compact = re.sub(r"\s+", "", normalized)
    return words + [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]


def hash_features(text: str, dims: int = 2048) -> dict[int, float]:
    vector: dict[int, float] = {}
    for feature in features(text):
        slot = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big") % dims
        vector[slot] = vector.get(slot, 0.0) + 1.0
    return vector


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def skill_retrieval_text(content: str, folder_name: str) -> str:
    """Use the evaluator's name, description, and when-to-use fields."""
    name = folder_name
    match = re.search(r"(?mi)^name\s*:\s*(.+?)\s*$", content)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
    description = ""
    match = re.search(r"(?mi)^description\s*:\s*(.+?)\s*$", content)
    if match:
        description = match.group(1).strip().strip('"').strip("'")
    when_to_use = ""
    match = re.search(r"(?ims)^##\s*when[\s-]*to[\s-]*use\s*\n(.*?)(?=^##\s|\Z)", content)
    if match:
        when_to_use = match.group(1).strip()
    return "\n".join(part for part in (name, description, when_to_use) if part)


def retrieve_skill(skillbank: Path, question: str) -> dict[str, Any]:
    if not skillbank.is_dir():
        raise NotADirectoryError(f"Skill bank not found: {skillbank}")
    query = hash_features(question)
    candidates: list[tuple[str, str, float]] = []
    for skill_path in sorted(skillbank.glob("*/SKILL.md")):
        content = skill_path.read_text(encoding="utf-8").strip()
        if content:
            score = cosine(query, hash_features(skill_retrieval_text(content, skill_path.parent.name)))
            candidates.append((skill_path.parent.name, content, score))
    if not candidates:
        raise RuntimeError(f"No SKILL.md files found in: {skillbank}")
    name, content, score = max(candidates, key=lambda item: (item[2], item[0]))
    # Always use the single highest-scoring skill; do not apply a similarity cutoff.
    return {"name": name, "content": content, "score": score, "used": True}


def load_runtime(config_path: Path) -> tuple[ChatOpenAI, dict[str, dict[str, str]]]:
    """Read the local LLM and externally started HTTP MCP services from config."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["models"][0]
    llm_kwargs: dict[str, Any] = {
        "model": model["model_name"],
        "api_key": model.get("api_key", ""),
        "base_url": model["client_args"]["base_url"],
        "temperature": 0.1,
        "timeout": 600,
        "max_retries": 2,
    }
    if "generate_args" in model:
        llm_kwargs["extra_body"] = model["generate_args"]
    mcp_servers: dict[str, dict[str, str]] = {}
    for name, server in config["mcpServers"].items():
        transport = server.get("transport", "stdio")
        if transport not in {"http", "streamable_http"}:
            raise ValueError(f"{name} uses {transport!r}; start the HTTP MCP services with run_tools_eb.sh.")
        mcp_servers[name] = {"transport": transport, "url": server["url"]}
    return ChatOpenAI(**llm_kwargs), mcp_servers


def normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or getattr(message, "additional_kwargs", {}).get("tool_calls", [])
    records = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = call.get("args", call.get("arguments", function.get("arguments", {})))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        records.append({"id": call.get("id", f"tool_call_{index}"), "name": call.get("name", function.get("name")), "arguments": arguments})
    return records


def serialize_message(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", "")
    try:
        json.dumps(content)
    except TypeError:
        content = str(content)
    record: dict[str, Any] = {"type": getattr(message, "type", type(message).__name__), "name": getattr(message, "name", None), "content": content}
    calls = normalize_tool_calls(message)
    if calls:
        record["tool_calls"] = calls
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        record["tool_call_id"] = tool_call_id
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Earth-Bench task with live MCP tools.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--skillbank", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--skill-top-k", type=int, choices=[1], default=1,
                        help="Retrieve and use the single highest-scoring skill.")
    parser.add_argument("--timeout", type=float, default=900, help="Whole-task timeout in seconds.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--use-skill", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


async def close_client(client: MultiServerMCPClient) -> None:
    closer = getattr(client, "close", None)
    if closer is not None:
        result = closer()
        if inspect.isawaitable(result):
            await result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    skillbank = Path(args.skillbank).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if args.max_rounds < 1 or args.timeout <= 0:
        raise ValueError("max-rounds and timeout must be positive")

    retrieved = retrieve_skill(skillbank, args.question) if args.use_skill else None
    if retrieved:
        print(f"[skill] top1={retrieved['name']} | cosine={retrieved['score']:.4f}")
    skill_context = ""
    if retrieved and retrieved["used"]:
        skill_context = ("\n\n## Retrieved Skill\nUse this as procedural guidance. Do not reuse sample-specific paths or values; use the current task and tool observations.\n\n" + retrieved["content"])
    user_query = (f"{SYSTEM_PROMPT}{skill_context}\n\nTask: {args.question}\nInput image: {image_path}\nInput directory: {image_path.parent}\nUse the image above to complete the task.")

    trace: dict[str, Any] = {
        "question": args.question,
        "image_path": str(image_path),
        "image_directory": str(image_path.parent),
        "skill": None if retrieved is None else {key: retrieved[key] for key in ("name", "score", "used")},
        "config_path": str(config_path),
        "messages": [], "final_answer": None, "error": None,
    }
    llm, mcp_servers = load_runtime(config_path)
    client = MultiServerMCPClient(mcp_servers)
    try:
        tools = await client.get_tools()
        print(f"[mcp] loaded {len(tools)} tools from {len(mcp_servers)} services")
        agent = create_react_agent(llm, tools)
        response = await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content=user_query)]}, config={"recursion_limit": args.max_rounds * 2 + 10}),
            timeout=args.timeout,
        )
        trace["messages"] = [serialize_message(message) for message in response.get("messages", [])]
        for message in reversed(response.get("messages", [])):
            if getattr(message, "type", None) == "ai" and str(getattr(message, "content", "")).strip():
                trace["final_answer"] = str(message.content).strip()
                break
        print("\n[final answer]\n" + (trace["final_answer"] or "No final answer produced."))
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {trace['error']}")
        raise
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[trace] saved to {output_path}")
        await close_client(client)
    return trace


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
