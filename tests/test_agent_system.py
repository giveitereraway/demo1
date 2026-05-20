from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from agent_system.commands import (
    Eval1v1Config,
    HumanLoopConfig,
    TrainConfig,
    VisualizeConfig,
    _path,
    build_eval_1v1_command,
    build_human_loop_command,
    build_train_command,
    build_visualize_command,
)
from agent_system.executor import run_command
from agent_system.result_analysis import infer_result_dir
from agent_system.routing import keyword_route, parse_route_json
from agent_system.settings import AgentSettings
from agent_system.settings import REPO_ROOT


def test_keyword_route_eval() -> None:
    decision = keyword_route("帮我评估两个 actor 的 1v1 胜率")
    assert decision.route == "evaluate_1v1"


def test_parse_route_json_strips_markdown_like_text() -> None:
    decision = parse_route_json('```json\n{"route":"train","confidence":0.9,"reason":"训练","extracted":{}}\n```')
    assert decision.route == "train"
    assert decision.confidence == 0.9


def test_build_train_command_uses_argument_list() -> None:
    spec = build_train_command(
        TrainConfig(
            scenario_name="1v1/NoWeapon/HierarchySelfplay",
            experiment_name="unit_train",
            cuda=False,
            use_wandb=False,
        )
    )
    assert spec.command[0] == AgentSettings.load().runtime_python
    assert "--scenario-name" in spec.command
    assert "1v1/NoWeapon/HierarchySelfplay" in spec.command
    assert "--clip-param" in spec.command
    assert "--clip-params" not in spec.command


def test_build_eval_command_exposes_cli_contract() -> None:
    spec = build_eval_1v1_command(
        Eval1v1Config(
            actor_a_path="envs/JSBSim/model/actor_latest.pt",
            actor_b_path="envs/JSBSim/model/actor_latest.pt",
            num_episodes=1,
            save_acmi=False,
        )
    )
    assert "--eval-scenario-name" in spec.command
    assert "--actor-a-scenario-name" in spec.command
    assert "--actor-b-scenario-name" in spec.command
    assert "--save-acmi" in spec.command
    assert "false" in spec.command
    assert spec.validation_errors == []


def test_path_normalization_accepts_slashes_quotes_and_reports_missing() -> None:
    normalized = _path('"envs\\JSBSim/model\\actor_heading.pt"')
    assert normalized == (REPO_ROOT / "envs" / "JSBSim" / "model" / "actor_heading.pt").resolve()

    spec = build_eval_1v1_command(
        Eval1v1Config(
            actor_a_path="scripts\\results/missing/filesl/actor_latest.pt",
            actor_b_path="envs/JSBSim/model/actor_latest.pt",
            lowlevel_actor_path="envs\\JSBSim/model\\actor_heading.pt",
        )
    )
    assert any("Actor A 文件不存在" in item for item in spec.validation_errors)


def test_build_human_and_visualize_commands() -> None:
    human = build_human_loop_command(HumanLoopConfig(mode="shoot_1v1", cuda=False))
    assert "human_shoot_1v1.py" in human.preview()
    assert "--use-prior" in human.command

    visualize = build_visualize_command(VisualizeConfig(model_dir="scripts/results/demo"))
    assert "render_jsbsim.py" in visualize.preview()
    assert "--model-dir" in visualize.command


def test_infer_result_dir_from_json_output() -> None:
    result_dir = infer_result_dir('{"output_dir": "E:/clone/demo1/experiments/results/x"}')
    assert result_dir is not None
    assert result_dir.as_posix().endswith("/experiments/results/x")


def test_run_command_without_shell() -> None:
    from agent_system.commands import CommandSpec

    result = run_command(CommandSpec("unit", [sys.executable, "-c", "print('ok')"]))
    assert result.ok
    assert "ok" in result.output


def test_rag_adapter_uses_runtime_python(monkeypatch, tmp_path) -> None:
    from agent_system import rag_adapter

    rag_root = tmp_path / "fake_rag_root"
    kb_dir = rag_root / "vector_store" / "faiss"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "index.faiss").write_bytes(b"fake")
    (kb_dir / "documents.jsonl").write_text("{}", encoding="utf-8")
    (kb_dir / "manifest.json").write_text(json.dumps({"dimension": 4096}), encoding="utf-8")
    settings = AgentSettings(siliconflow_api_key="", rag_project_root=rag_root)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"answer": "ok", "sources": "", "retrieval_json": "{}"}),
            stderr="",
        )

    monkeypatch.setattr(rag_adapter.subprocess, "run", fake_run)
    response = rag_adapter.answer_with_rag("测试问题", settings=settings, knowledge_base_dir=kb_dir)

    assert response.answer == "ok"
    assert captured["command"][0] == settings.runtime_python
    assert captured["command"][1:] == ["-m", "agent_system.rag_worker"]
    assert captured["kwargs"]["shell"] is False
    payload = json.loads(captured["kwargs"]["input"])
    assert payload["knowledge_base_dir"] == str(kb_dir.resolve())
    assert payload["embedding_dimensions"] == 4096
    assert captured["kwargs"]["env"]["AGENTIC_RAG_VECTOR_STORE_DIR"] == str(kb_dir.resolve())
    assert captured["kwargs"]["env"]["SILICONFLOW_EMBEDDING_DIMENSIONS"] == "4096"


def test_rag_knowledge_base_discovery(tmp_path) -> None:
    from agent_system.rag_adapter import discover_knowledge_bases, resolve_knowledge_base_dir, validate_knowledge_base_dir

    rag_root = tmp_path / "rag"
    kb_dir = rag_root / "vector_store" / "faiss"
    kb_dir.mkdir(parents=True)
    (kb_dir / "index.faiss").write_bytes(b"fake")
    (kb_dir / "documents.jsonl").write_text("{}", encoding="utf-8")
    (kb_dir / "manifest.json").write_text(
        json.dumps({"doc_count": 3, "dimension": 4096}, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = AgentSettings(siliconflow_api_key="", rag_project_root=rag_root)

    assert resolve_knowledge_base_dir("vector_store/faiss", settings) == kb_dir.resolve()
    assert validate_knowledge_base_dir(kb_dir) == []
    options = discover_knowledge_bases(settings)
    assert len(options) == 1
    assert options[0].path == kb_dir.resolve()
    assert "3 chunks" in options[0].label
