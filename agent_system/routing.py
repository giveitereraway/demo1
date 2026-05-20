from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .llm import LLMMessage, SiliconFlowClient


RouteType = Literal["train", "evaluate_1v1", "human_loop", "visualize", "rag_qa", "unknown"]
ROUTE_TYPES = {"train", "evaluate_1v1", "human_loop", "visualize", "rag_qa", "unknown"}


@dataclass
class RouteDecision:
    route: RouteType
    confidence: float = 0.0
    reason: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "extracted": self.extracted,
        }


ROUTER_SYSTEM_PROMPT = """你是飞行器自主决策实验平台的任务管理 Agent。
请把用户输入路由到且只路由到以下流程之一：
train, evaluate_1v1, human_loop, visualize, rag_qa, unknown。
只返回 JSON，不要返回 Markdown。JSON 字段为 route、confidence、reason、extracted。
LLM 只负责理解和编排，不能直接控制低层飞行动作。"""


def parse_route_json(text: str) -> RouteDecision:
    """解析 LLM 返回的 JSON；失败时交给关键词兜底。"""
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    route = str(parsed.get("route", "unknown")).strip()
    if route not in ROUTE_TYPES:
        route = "unknown"
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    reason = str(parsed.get("reason", ""))
    extracted = parsed.get("extracted", {})
    if not isinstance(extracted, dict):
        extracted = {}
    return RouteDecision(route=route, confidence=confidence, reason=reason, extracted=extracted)  # type: ignore[arg-type]


def keyword_route(user_input: str) -> RouteDecision:
    text = user_input.lower()
    if any(word in user_input for word in ("训练", "继续训练", "自博弈")) or "train" in text:
        return RouteDecision("train", 0.55, "关键词匹配到训练流程。")
    if any(word in user_input for word in ("评估", "胜率", "对比", "1v1")) or "eval" in text:
        return RouteDecision("evaluate_1v1", 0.55, "关键词匹配到 1v1 评估流程。")
    if any(word in user_input for word in ("人机", "手动", "操控", "键盘")) or "human" in text:
        return RouteDecision("human_loop", 0.55, "关键词匹配到人机交互流程。")
    if any(word in user_input for word in ("可视化", "轨迹", "tacview", "渲染")) or "render" in text:
        return RouteDecision("visualize", 0.55, "关键词匹配到 Tacview 可视化流程。")
    if any(word in user_input for word in ("问答", "知识库", "rag", "文档", "解释")):
        return RouteDecision("rag_qa", 0.5, "关键词匹配到 RAG 问答流程。")
    return RouteDecision("unknown", 0.2, "未识别出明确流程。")


def route_user_input(user_input: str, client: SiliconFlowClient | None = None) -> RouteDecision:
    if not user_input.strip():
        return RouteDecision("unknown", 0.0, "输入为空。")
    if client is None:
        return keyword_route(user_input)
    try:
        content = client.chat(
            [
                LLMMessage("system", ROUTER_SYSTEM_PROMPT),
                LLMMessage("user", user_input),
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return parse_route_json(content)
    except Exception as exc:
        fallback = keyword_route(user_input)
        fallback.reason = f"{fallback.reason} LLM 路由失败，已使用关键词兜底: {exc}"
        return fallback


class SimpleTaskGraph:
    """LangGraph 不可用时的等价轻量入口。"""

    def __init__(self, client: SiliconFlowClient | None = None) -> None:
        self.client = client

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        user_input = str(state.get("user_input", ""))
        decision = route_user_input(user_input, self.client)
        return {**state, "route_decision": decision.as_dict()}


def build_task_graph(client: SiliconFlowClient | None = None):
    """优先构建 LangGraph；缺依赖时返回同接口轻量图。"""
    try:
        from langgraph.graph import END, StateGraph
        from typing_extensions import TypedDict
    except Exception:
        return SimpleTaskGraph(client)

    class TaskState(TypedDict, total=False):
        user_input: str
        route_decision: dict[str, Any]

    def route_node(state: TaskState) -> TaskState:
        decision = route_user_input(str(state.get("user_input", "")), client)
        return {"route_decision": decision.as_dict()}

    graph = StateGraph(TaskState)
    graph.add_node("route", route_node)
    graph.set_entry_point("route")
    graph.add_edge("route", END)
    return graph.compile()
