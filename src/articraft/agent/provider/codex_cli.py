from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from articraft.agent._child_process import child_environment
from articraft.errors import ModelError
from articraft.settings import DEFAULT_CODEX_MODEL, Settings, get_settings

_CONTEXT_WINDOW_TOKENS = 272_000
_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def context_window_tokens_for(model: str) -> int | None:
    return _CONTEXT_WINDOW_TOKENS if model == DEFAULT_CODEX_MODEL else None


@dataclass(frozen=True)
class CodexCliExecResult:
    returncode: int
    stdout: str
    stderr: str
    last_message: str


CodexCliRunner = Callable[[list[str], str, float, Path], Awaitable[CodexCliExecResult]]


class CodexCliModel:
    """Use Codex CLI as the model behind the normal Articraft agent loop."""

    supports_images = True

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        runner: CodexCliRunner | None = None,
    ):
        self.config = settings or get_settings()
        self._runner = runner or _run_codex_exec
        if runner is None and shutil.which(self.config.codex_cli_bin) is None:
            raise ValueError(
                "Codex CLI provider requires the `codex` executable. "
                "Install and log in to Codex CLI, or set ARTICRAFT_CODEX_CLI_BIN."
            )

    @property
    def context_window_tokens(self) -> int:
        return _CONTEXT_WINDOW_TOKENS

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self._query(messages, tools=tools or [], summary=False)

    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        del max_output_tokens
        return await self._query(messages, tools=[], summary=True)

    async def close(self) -> None:
        return None

    async def _query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        summary: bool,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="articraft-codex-") as raw_tmp:
            tmp_dir = Path(raw_tmp)
            schema_path = tmp_dir / "assistant-turn.schema.json"
            output_path = tmp_dir / "assistant-turn.json"
            schema_path.write_text(json.dumps(_OUTPUT_SCHEMA), encoding="utf-8")
            images, image_names = _materialize_images(messages, tmp_dir)
            prompt = _render_prompt(messages, tools, image_names, summary=summary)
            command = self._command(schema_path, output_path, images)
            result = await self._runner(
                command,
                prompt,
                self.config.codex_cli_timeout_seconds,
                output_path,
            )

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise ModelError(f"Codex CLI exited with status {result.returncode}: {detail[-4000:]}")
        payload = _parse_response(result.last_message)
        text = str(payload.get("text") or "")
        tool_calls = _tool_calls(payload.get("tool_calls"))
        if summary and not text.strip():
            raise ModelError("Codex CLI summary response did not contain text")
        if not summary and not text.strip() and not tool_calls:
            raise ModelError("Codex CLI response did not contain text or tool calls")

        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": _token_usage(result.stdout, result.stderr),
            "cost": 0.0,
            "response": payload,
        }

    def _command(
        self,
        schema_path: Path,
        output_path: Path,
        images: list[Path],
    ) -> list[str]:
        command = [
            self.config.codex_cli_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-C",
            str(Path.cwd().resolve()),
        ]
        if self.config.codex_model != DEFAULT_CODEX_MODEL:
            command.extend(["--model", self.config.codex_model])
        for image in images:
            command.extend(["--image", str(image)])
        effort = self.config.selected_reasoning_effort.strip()
        if effort:
            command.extend(["-c", f'model_reasoning_effort="{effort}"'])
        command.append("-")
        return command


async def _run_codex_exec(
    command: list[str],
    prompt: str,
    timeout_seconds: float,
    output_path: Path,
) -> CodexCliExecResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_environment(),
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as exc:
        raise ModelError("Codex CLI executable was not found") from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        await _stop_process(process)
        raise ModelError(f"Codex CLI timed out after {timeout_seconds:g}s") from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise

    last_message = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    return CodexCliExecResult(
        returncode=int(process.returncode or 0),
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        last_message=last_message,
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


def _render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    image_names: dict[str, str],
    *,
    summary: bool,
) -> str:
    task = (
        "Summarize the supplied conversation into a compact checkpoint. Return the summary in "
        "text and return an empty tool_calls list."
        if summary
        else (
            "Return exactly one assistant turn. Use tool_calls when Articraft should act next; "
            "return text without tool_calls only when concluding or reporting a blocker."
        )
    )
    return "\n\n".join(
        [
            "You are the Codex CLI model transport for Articraft's existing agent.",
            (
                "Do not edit files, run shell commands, or use Codex tools. Articraft will execute "
                "the declared function calls, compile the model, and preserve the run record."
            ),
            task,
            "<available_tools>\n"
            + json.dumps(_tool_reference(tools), indent=2, ensure_ascii=False)
            + "\n</available_tools>",
            "<conversation>\n"
            + json.dumps(
                _conversation_reference(messages, image_names),
                indent=2,
                ensure_ascii=False,
            )
            + "\n</conversation>",
            (
                "Each tool call must name an available tool and encode its arguments as a JSON "
                "object string. Return JSON matching the required output schema."
            ),
        ]
    )


def _tool_reference(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters") or {"type": "object"},
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _conversation_reference(
    messages: list[dict[str, Any]],
    image_names: dict[str, str],
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            rendered.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("call_id") or ""),
                    "output": _content_reference(message.get("output"), image_names),
                }
            )
            continue
        item: dict[str, Any] = {
            "role": str(message.get("role") or ""),
            "content": _content_reference(message.get("content"), image_names),
        }
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            item["tool_calls"] = calls
        rendered.append(item)
    return rendered


def _content_reference(content: Any, image_names: dict[str, str]) -> Any:
    if not isinstance(content, list):
        return content
    rendered: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"}:
            rendered.append({"type": "text", "text": str(part.get("text") or "")})
        elif part.get("type") in {"input_image", "image"}:
            url = str(part.get("image_url") or "")
            rendered.append(
                {
                    "type": "image",
                    "attachment": image_names.get(url, "unavailable"),
                    "detail": str(part.get("detail") or ""),
                }
            )
        else:
            rendered.append(dict(part))
    return rendered


def _materialize_images(
    messages: list[dict[str, Any]],
    directory: Path,
) -> tuple[list[Path], dict[str, str]]:
    paths: list[Path] = []
    names: dict[str, str] = {}
    for message in messages:
        for field in ("content", "output"):
            value = message.get(field)
            if not isinstance(value, list):
                continue
            for part in value:
                if not isinstance(part, dict) or part.get("type") not in {
                    "input_image",
                    "image",
                }:
                    continue
                url = str(part.get("image_url") or "")
                if not url or url in names:
                    continue
                mime_type, data = _decode_data_url(url)
                digest = hashlib.sha256(data).hexdigest()[:12]
                path = directory / f"image-{len(paths) + 1}-{digest}{_MIME_SUFFIXES[mime_type]}"
                path.write_bytes(data)
                paths.append(path)
                names[url] = path.name
    return paths, names


def _decode_data_url(url: str) -> tuple[str, bytes]:
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", url, flags=re.DOTALL)
    if match is None or match.group(1) not in _MIME_SUFFIXES:
        raise ModelError("Codex CLI image input must be a PNG, JPEG, or WebP data URL")
    try:
        return match.group(1), base64.b64decode(match.group(2), validate=True)
    except ValueError as exc:
        raise ModelError("Codex CLI image input contains invalid base64 data") from exc


def _parse_response(raw: str) -> dict[str, Any]:
    if not raw.strip():
        raise ModelError("Codex CLI did not write an assistant response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelError("Codex CLI assistant response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelError("Codex CLI assistant response must be a JSON object")
    return payload


def _tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ModelError("Codex CLI tool_calls must be a list")
    calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            raise ModelError("Codex CLI returned an invalid tool call")
        arguments = item.get("arguments")
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            raise ModelError("Codex CLI tool arguments were not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ModelError("Codex CLI tool arguments must be a JSON object")
        calls.append(
            {
                "id": f"call_codex_{uuid.uuid4().hex}",
                "name": str(item["name"]),
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return calls


def _token_usage(stdout: str, stderr: str) -> dict[str, int]:
    matches = re.findall(
        r"tokens used\s*(?:\r?\n|\s)+([0-9][0-9,]*)",
        "\n".join((stdout, stderr)),
        flags=re.IGNORECASE,
    )
    return {"total_tokens": int(matches[-1].replace(",", ""))} if matches else {}


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        "type": "string",
                        "description": "JSON object string containing the tool arguments.",
                    },
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["text", "tool_calls"],
}


__all__ = ["CodexCliExecResult", "CodexCliModel", "context_window_tokens_for"]
