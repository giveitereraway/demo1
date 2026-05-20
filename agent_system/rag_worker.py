from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _patch_dataclass_slots_for_legacy_python() -> None:
    """兼容 Python 3.9：忽略外部 RAG 代码中的新版 dataclass 参数。"""
    if sys.version_info >= (3, 10):
        return

    import dataclasses
    import inspect

    try:
        parameters = inspect.signature(dataclasses.dataclass).parameters
    except (TypeError, ValueError):
        return
    if "slots" in parameters:
        return

    original_dataclass = dataclasses.dataclass
    unsupported_keys = {"match_args", "kw_only", "slots", "weakref_slot"} - set(parameters)

    def compatible_dataclass(cls=None, /, **kwargs):
        for key in unsupported_keys:
            kwargs.pop(key, None)
        return original_dataclass(cls, **kwargs)

    dataclasses.dataclass = compatible_dataclass


def _patch_typing_for_legacy_python() -> None:
    """兼容 Python 3.9：把 typing_extensions 中的新版类型补到 typing。"""
    if sys.version_info >= (3, 11):
        return

    import typing

    try:
        import typing_extensions
    except ImportError:
        return

    for name in ("NotRequired", "Required", "Self", "TypeAlias"):
        if not hasattr(typing, name) and hasattr(typing_extensions, name):
            setattr(typing, name, getattr(typing_extensions, name))


def _add_import_paths(rag_project_root: Path) -> None:
    """把外部 RAG 项目的源码路径加入当前子进程导入路径。"""
    for path in (rag_project_root / "src", rag_project_root):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def run(payload: dict[str, Any]) -> dict[str, str]:
    rag_project_root = Path(payload["rag_project_root"])
    _add_import_paths(rag_project_root)
    _patch_typing_for_legacy_python()
    _patch_dataclass_slots_for_legacy_python()

    embedding_dimensions = payload.get("embedding_dimensions")
    if isinstance(embedding_dimensions, int) and embedding_dimensions > 0:
        os.environ["SILICONFLOW_EMBEDDING_DIMENSIONS"] = str(embedding_dimensions)

    from langchain_core.messages import SystemMessage

    from agentic_rag.rag_agent import (
        ANSWER_SYSTEM_PROMPT,
        HybridRerankRetriever,
        RagAgentConfig,
        build_multimodal_answer_message,
        collect_image_contexts,
        create_chat_model,
        create_reranker,
        format_sources,
        retrieval_result_to_json,
    )
    from agentic_rag.vector_store import VectorStoreConfig, load_vector_store

    knowledge_base_dir = payload.get("knowledge_base_dir")
    config = RagAgentConfig.from_env(persist_dir=knowledge_base_dir)
    config.chat_model = payload["chat_model"]
    config.base_url = payload["base_url"]
    if hasattr(config, "api_key"):
        config.api_key = payload.get("api_key") or config.api_key
    if hasattr(config, "rerank_api_key"):
        config.rerank_api_key = payload.get("api_key") or config.rerank_api_key

    # Web Agent 场景下用用户原问题直接检索，避免 LLM 先改写工具参数导致查询不稳定。
    vector_config = VectorStoreConfig.from_env(persist_dir=config.persist_dir)
    vector_store = load_vector_store(vector_config)
    reranker = create_reranker(config) if config.rerank_enabled else None
    retriever = HybridRerankRetriever(
        vector_store,
        reranker=reranker,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight,
    )
    retrieval_result = retriever.retrieve(
        payload["question"],
        k=config.k,
        vector_k=config.vector_k,
        keyword_k=config.keyword_k,
        rerank_enabled=config.rerank_enabled,
    )
    images = collect_image_contexts(
        retrieval_result.chunks,
        max_images=config.max_images,
        image_detail=config.image_detail,
        workspace_root=config.workspace_root,
    )
    answer_message = build_multimodal_answer_message(payload["question"], retrieval_result, images)
    response = create_chat_model(config).invoke([SystemMessage(content=ANSWER_SYSTEM_PROMPT), answer_message])
    return {
        "answer": _message_content_to_text(getattr(response, "content", "")),
        "sources": format_sources(retrieval_result),
        "retrieval_json": retrieval_result_to_json(retrieval_result),
    }


def _message_content_to_text(content: Any) -> str:
    """把 LangChain 消息内容统一转成页面可展示的文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = run(payload)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "python": sys.executable,
                    "python_version": sys.version,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
