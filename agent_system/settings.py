from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_AGENT_MODEL = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_RAG_PROJECT_ROOT = Path(r"E:\AI_practice\Agentic_RAG")
DEFAULT_AGENT_PYTHON_ENV = Path(r"C:\ProgramData\anaconda3\envs\jsbsim")


def load_dotenv(path: Path | None = None) -> None:
    """轻量加载 .env，避免为启动器额外引入依赖。"""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AgentSettings:
    siliconflow_api_key: str
    siliconflow_base_url: str = DEFAULT_SILICONFLOW_BASE_URL
    agent_chat_model: str = DEFAULT_AGENT_MODEL
    rag_project_root: Path = DEFAULT_RAG_PROJECT_ROOT
    python_env: Path = DEFAULT_AGENT_PYTHON_ENV
    python_executable: Path = DEFAULT_AGENT_PYTHON_ENV / "python.exe"
    repo_root: Path = REPO_ROOT

    @classmethod
    def load(cls) -> "AgentSettings":
        load_dotenv()
        python_env = Path(os.getenv("AGENT_PYTHON_ENV", str(DEFAULT_AGENT_PYTHON_ENV)))
        python_executable = Path(os.getenv("AGENT_PYTHON_EXECUTABLE", str(python_env / "python.exe")))
        return cls(
            siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
            agent_chat_model=os.getenv("AGENT_CHAT_MODEL", DEFAULT_AGENT_MODEL),
            rag_project_root=Path(os.getenv("RAG_PROJECT_ROOT", str(DEFAULT_RAG_PROJECT_ROOT))),
            python_env=python_env,
            python_executable=python_executable,
        )

    @property
    def rag_src_path(self) -> Path:
        return self.rag_project_root / "src"

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.siliconflow_api_key.strip())

    @property
    def runtime_python(self) -> str:
        """返回训练、评估、人机交互和可视化脚本使用的 Python 解释器。"""
        return str(self.python_executable) if self.python_executable.exists() else sys.executable
