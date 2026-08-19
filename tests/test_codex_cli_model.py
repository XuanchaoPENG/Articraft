from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest
from harness import GOOD_MAIN_PY, WarmEnvironment

from articraft.agent import Agent
from articraft.agent.provider import create_model
from articraft.agent.provider.codex_cli import CodexCliExecResult, CodexCliModel
from articraft.errors import ModelError
from articraft.settings import DEFAULT_CODEX_MODEL, Settings


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def test_codex_cli_model_returns_agent_tool_calls_and_usage() -> None:
    captured: dict[str, Any] = {}

    async def runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        captured.update(command=command, prompt=prompt, timeout=timeout_seconds)
        return CodexCliExecResult(
            0,
            "tokens used\n1,234\n",
            "",
            json.dumps(
                {
                    "text": "",
                    "tool_calls": [
                        {
                            "name": "write",
                            "arguments": json.dumps({"path": "main.py", "content": "print('ok')"}),
                        }
                    ],
                }
            ),
        )

    settings = Settings(
        provider="codex-cli",
        codex_model="gpt-test",
        codex_cli_timeout_seconds=45,
        openai_reasoning_effort="medium",
    )
    model = CodexCliModel(settings, runner=runner)
    tool = {
        "type": "function",
        "name": "write",
        "description": "Write a file",
        "parameters": {"type": "object", "properties": {}},
    }

    result = run(model.query([{"role": "user", "content": "make a hinge"}], tools=[tool]))

    assert result["text"] == ""
    assert result["token_usage"] == {"total_tokens": 1234}
    assert result["cost"] == 0.0
    assert result["tool_calls"][0]["name"] == "write"
    assert json.loads(result["tool_calls"][0]["arguments"])["path"] == "main.py"
    command = captured["command"]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--model") + 1] == "gpt-test"
    assert 'model_reasoning_effort="medium"' in command
    assert captured["timeout"] == 45
    assert '"name": "write"' in captured["prompt"]
    assert "Do not edit files, run shell commands" in captured["prompt"]


def test_codex_cli_uses_configured_default_model_without_model_flag() -> None:
    async def runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        assert "--model" not in command
        return CodexCliExecResult(0, "", "", '{"text":"done","tool_calls":[]}')

    model = CodexCliModel(
        Settings(provider="codex-cli", codex_model=DEFAULT_CODEX_MODEL),
        runner=runner,
    )

    assert run(model.query([{"role": "user", "content": "task"}]))["text"] == "done"


def test_codex_cli_materializes_message_images() -> None:
    captured: dict[str, Any] = {}
    encoded = base64.b64encode(b"png-data").decode()

    async def runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        image_path = Path(command[command.index("--image") + 1])
        captured["image"] = image_path.read_bytes()
        captured["prompt"] = prompt
        return CodexCliExecResult(0, "", "", '{"text":"done","tool_calls":[]}')

    model = CodexCliModel(Settings(provider="codex-cli"), runner=runner)
    result = run(
        model.query(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "copy this"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded}",
                            "detail": "original",
                        },
                    ],
                }
            ]
        )
    )

    assert result["text"] == "done"
    assert captured["image"] == b"png-data"
    assert "image-1-" in captured["prompt"]
    assert encoded not in captured["prompt"]


def test_codex_cli_summary_uses_the_existing_compaction_protocol() -> None:
    async def runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        assert "Summarize the supplied conversation" in prompt
        assert "<available_tools>\n[]" in prompt
        return CodexCliExecResult(
            0,
            "tokens used\n20",
            "",
            '{"text":"checkpoint","tool_calls":[]}',
        )

    model = CodexCliModel(Settings(provider="codex-cli"), runner=runner)

    result = run(
        model.summarize_context(
            [{"role": "user", "content": "old work"}],
            max_output_tokens=100,
        )
    )

    assert result["text"] == "checkpoint"
    assert result["token_usage"] == {"total_tokens": 20}


def test_codex_cli_surfaces_process_and_output_errors() -> None:
    async def failed_runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        return CodexCliExecResult(2, "", "not logged in", "")

    model = CodexCliModel(Settings(provider="codex-cli"), runner=failed_runner)
    with pytest.raises(ModelError, match="not logged in"):
        run(model.query([{"role": "user", "content": "task"}]))


def test_factory_creates_codex_cli_model(monkeypatch) -> None:
    monkeypatch.setattr("articraft.agent.provider.codex_cli.shutil.which", lambda binary: binary)

    model = create_model(Settings(provider="codex-cli"))

    assert isinstance(model, CodexCliModel)


def test_codex_cli_reports_a_missing_executable(monkeypatch) -> None:
    monkeypatch.setattr("articraft.agent.provider.codex_cli.shutil.which", lambda binary: None)

    with pytest.raises(ValueError, match="requires the `codex` executable"):
        CodexCliModel(Settings(provider="codex-cli"))


def test_codex_cli_model_runs_through_the_existing_agent(tmp_path: Path) -> None:
    responses = [
        {
            "text": "",
            "tool_calls": [
                {
                    "name": "write",
                    "arguments": json.dumps({"path": "main.py", "content": GOOD_MAIN_PY}),
                }
            ],
        },
        {
            "text": "",
            "tool_calls": [{"name": "compile", "arguments": "{}"}],
        },
        {"text": "done", "tool_calls": []},
    ]

    async def runner(
        command: list[str],
        prompt: str,
        timeout_seconds: float,
        output_path: Path,
    ) -> CodexCliExecResult:
        return CodexCliExecResult(0, "", "", json.dumps(responses.pop(0)))

    model = CodexCliModel(Settings(provider="codex-cli"), runner=runner)
    workspace = WarmEnvironment(output_dir=tmp_path / "runs")

    result = run(Agent(model, workspace).run("make a box"))

    assert result["status"] == "success"
    assert result["message"] == "done"
    assert responses == []
