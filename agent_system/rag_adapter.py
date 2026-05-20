from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .settings import AgentSettings, REPO_ROOT


@dataclass
class RagResponse:
    answer: str
    sources: str
    retrieval_json: str


@dataclass(frozen=True)
class KnowledgeBaseOption:
    name: str
    path: Path
    doc_count: int | None = None
    dimension: int | None = None
    index_content_version: str | None = None

    @property
    def label(self) -> str:
        parts = [self.name]
        details = []
        if self.doc_count is not None:
            details.append(f"{self.doc_count} chunks")
        if self.dimension is not None:
            details.append(f"dim {self.dimension}")
        if self.index_content_version:
            details.append(self.index_content_version)
        if details:
            parts.append(f"（{', '.join(details)}）")
        return "".join(parts)


def _path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def resolve_knowledge_base_dir(value: str | Path | None, settings: AgentSettings | None = None) -> Path:
    settings = settings or AgentSettings.load()
    text = _path_text(value)
    if not text:
        return (settings.rag_project_root / "vector_store" / "faiss").resolve()

    path = Path(text)
    if not path.is_absolute():
        path = settings.rag_project_root / path
    return path.resolve()


def validate_knowledge_base_dir(path: str | Path) -> list[str]:
    kb_dir = Path(path)
    errors: list[str] = []
    if not kb_dir.exists():
        return [f"知识库目录不存在：{kb_dir}"]
    if not kb_dir.is_dir():
        return [f"知识库路径不是目录：{kb_dir}"]
    if not (kb_dir / "index.faiss").exists():
        errors.append(f"缺少 FAISS 索引文件：{kb_dir / 'index.faiss'}")
    if not (kb_dir / "documents.jsonl").exists():
        errors.append(f"缺少文档索引文件：{kb_dir / 'documents.jsonl'}")
    return errors


def _read_manifest(kb_dir: Path) -> dict[str, object]:
    manifest_path = kb_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _relative_kb_name(path: Path, settings: AgentSettings) -> str:
    try:
        return path.relative_to(settings.rag_project_root).as_posix()
    except ValueError:
        return str(path)


def _knowledge_base_option(path: Path, settings: AgentSettings) -> KnowledgeBaseOption:
    manifest = _read_manifest(path)
    doc_count = manifest.get("doc_count")
    dimension = manifest.get("dimension")
    return KnowledgeBaseOption(
        name=_relative_kb_name(path, settings),
        path=path,
        doc_count=doc_count if isinstance(doc_count, int) else None,
        dimension=dimension if isinstance(dimension, int) else None,
        index_content_version=manifest.get("index_content_version") if isinstance(manifest.get("index_content_version"), str) else None,
    )


def _knowledge_base_embedding_dimensions(path: Path) -> int | None:
    manifest = _read_manifest(path)
    for key in ("dimensions", "dimension"):
        value = manifest.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def discover_knowledge_bases(settings: AgentSettings | None = None) -> list[KnowledgeBaseOption]:
    settings = settings or AgentSettings.load()
    roots = [
        settings.rag_project_root / "vector_store",
        settings.rag_project_root / "eval_vector_store",
    ]
    candidates = [
        resolve_knowledge_base_dir(os.getenv("AGENTIC_RAG_VECTOR_STORE_DIR"), settings),
        settings.rag_project_root / "vector_store" / "faiss",
    ]
    for root in roots:
        if not root.exists():
            continue
        for index_path in root.rglob("index.faiss"):
            candidates.append(index_path.parent)

    options: list[KnowledgeBaseOption] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen or validate_knowledge_base_dir(path):
            continue
        seen.add(path)
        options.append(_knowledge_base_option(path, settings))
    return options


def ensure_rag_import_path(settings: AgentSettings) -> None:
    """把外部 RAG 项目加入导入路径，不复制其内部实现。"""
    for path in (settings.rag_src_path, settings.rag_project_root):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def _format_worker_error(detail: str) -> str:
    error_payload = None
    for line in reversed(detail.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            error_payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(error_payload, dict):
        return detail

    error = str(error_payload.get("error", "")).strip()
    error_type = str(error_payload.get("error_type", "")).strip()
    python_version = str(error_payload.get("python_version", "")).splitlines()[0]
    if "dataclass() got an unexpected keyword argument 'slots'" in error:
        return (
            "外部 RAG 代码使用了 dataclass(slots=True)，当前 RAG 子进程 Python 不支持该参数。"
            f"子进程版本：{python_version or '未知'}。"
        )
    if error_type:
        return f"{error_type}: {error}"
    return error or detail


def answer_with_rag(
    question: str,
    settings: AgentSettings | None = None,
    knowledge_base_dir: str | Path | None = None,
) -> RagResponse:
    settings = settings or AgentSettings.load()
    resolved_kb_dir = resolve_knowledge_base_dir(knowledge_base_dir, settings)
    validation_errors = validate_knowledge_base_dir(resolved_kb_dir)
    if validation_errors:
        raise ValueError("知识库校验未通过：\n" + "\n".join(f"- {item}" for item in validation_errors))
    embedding_dimensions = _knowledge_base_embedding_dimensions(resolved_kb_dir)

    payload = {
        "question": question,
        "rag_project_root": str(settings.rag_project_root),
        "knowledge_base_dir": str(resolved_kb_dir),
        "embedding_dimensions": embedding_dimensions,
        "chat_model": settings.agent_chat_model,
        "base_url": settings.siliconflow_base_url,
        "api_key": settings.siliconflow_api_key,
    }
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(settings.rag_src_path), str(settings.rag_project_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SILICONFLOW_API_KEY": settings.siliconflow_api_key,
            "SILICONFLOW_BASE_URL": settings.siliconflow_base_url,
            "SILICONFLOW_CHAT_MODEL": settings.agent_chat_model,
            "AGENTIC_RAG_VECTOR_STORE_DIR": str(resolved_kb_dir),
        }
    )
    if embedding_dimensions is not None:
        env["SILICONFLOW_EMBEDDING_DIMENSIONS"] = str(embedding_dimensions)

    completed = subprocess.run(
        [settings.runtime_python, "-m", "agent_system.rag_worker"],
        cwd=str(settings.rag_project_root if settings.rag_project_root.exists() else REPO_ROOT),
        env=env,
        input=json.dumps(payload, ensure_ascii=False),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"RAG 子进程执行失败：{_format_worker_error(detail)}")

    data = json.loads(completed.stdout)
    return RagResponse(
        answer=data.get("answer", ""),
        sources=data.get("sources", ""),
        retrieval_json=data.get("retrieval_json", "{}"),
    )
