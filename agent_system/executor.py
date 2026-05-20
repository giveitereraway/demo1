from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .commands import CommandSpec


@dataclass
class CommandResult:
    returncode: int
    output: str
    expected_output_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def stream_command(spec: CommandSpec) -> Iterator[str]:
    """逐行执行子进程输出；始终使用参数列表，不走 shell。"""
    process = subprocess.Popen(
        spec.command,
        cwd=str(spec.cwd),
        env=spec.merged_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip("\n")
    returncode = process.wait()
    if returncode != 0:
        yield f"[process exited with code {returncode}]"


def run_command(spec: CommandSpec, timeout: int | None = None) -> CommandResult:
    completed = subprocess.run(
        spec.command,
        cwd=str(spec.cwd),
        env=spec.merged_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=timeout,
    )
    return CommandResult(completed.returncode, completed.stdout, spec.expected_output_dir)
