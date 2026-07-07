"""OpenSandboxBackend 单元测试:内存假沙箱驱动三原语与原生 read,无需服务端。"""

from __future__ import annotations

import base64
import os
import posixpath
import shlex

import pytest

from deepagents.backends.protocol import FILE_NOT_FOUND, PERMISSION_DENIED
from deepagents.backends.sandbox import MAX_BINARY_BYTES, MAX_OUTPUT_BYTES, TRUNCATION_MSG
from deepagents_opensandbox import OpenSandboxBackend
from deepagents_opensandbox.backend import _classify, _exit_code, _output
from opensandbox.exceptions import SandboxApiException, SandboxException
from opensandbox.models.execd import Execution, ExecutionError, ExecutionLogs, OutputMessage


def _message(text: str, ts: int, *, is_error: bool = False) -> OutputMessage:
    return OutputMessage(text=text, timestamp=ts, is_error=is_error)


def _execution(*, stdout=(), stderr=(), exit_code=0, error=None) -> Execution:
    return Execution(
        logs=ExecutionLogs(stdout=list(stdout), stderr=list(stderr)),
        exit_code=exit_code,
        error=error,
    )


class FakeCommands:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox
        self.responder = None
        self.raise_exc: Exception | None = None
        self.last_command: str | None = None
        self.last_opts = None

    def run(self, command, *, opts=None, handlers=None) -> Execution:
        self.last_command = command
        self.last_opts = opts
        if self.raise_exc is not None:
            raise self.raise_exc
        if command.startswith("mkdir -p "):
            target = shlex.split(command[len("mkdir -p "):])[0]
            self._sandbox.dirs.add(target)
            return _execution(exit_code=0)
        if self.responder is not None:
            return self.responder(command, opts)
        return _execution(stdout=[_message("ok", 1)], exit_code=0)


class FakeFiles:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox
        self.strict_parents = False

    def write_file(self, path, data, *, encoding="utf-8", mode=755, owner=None, group=None) -> None:
        parent = posixpath.dirname(path)
        if self.strict_parents and parent not in ("", "/") and parent not in self._sandbox.dirs:
            raise SandboxApiException("No such file or directory", status_code=500)
        self._sandbox.files_store[path] = data if isinstance(data, bytes) else str(data).encode(encoding)

    def read_bytes(self, path, *, range_header=None, offset=None, limit=None) -> bytes:
        if path not in self._sandbox.files_store:
            raise SandboxApiException(f"file not found: {path}", status_code=404)
        return self._sandbox.files_store[path]


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-test-1") -> None:
        self.id = sandbox_id
        self.files_store: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.commands = FakeCommands(self)
        self.files = FakeFiles(self)
        self.killed = False
        self.closed = False

    def kill(self) -> None:
        self.killed = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def sandbox() -> FakeSandbox:
    return FakeSandbox()


@pytest.fixture
def backend(sandbox: FakeSandbox) -> OpenSandboxBackend:
    return OpenSandboxBackend(sandbox)  # type: ignore[arg-type]


def test_output_orders_by_timestamp():
    execution = _execution(
        stdout=[_message("first", 1), _message("third", 3)],
        stderr=[_message("second", 2, is_error=True)],
    )
    assert _output(execution) == "first\nsecond\nthird"


def test_output_empty_is_empty_string():
    assert _output(_execution()) == ""


def test_output_strips_trailing_newlines():
    execution = _execution(stdout=[_message("line\n", 1), _message("next\n", 2)])
    assert _output(execution) == "line\nnext"


def test_exit_code_prefers_explicit():
    assert _exit_code(_execution(exit_code=7)) == 7


def test_exit_code_infers_success_when_missing():
    assert _exit_code(_execution(exit_code=None)) == 0


def test_exit_code_infers_failure_from_error():
    err = ExecutionError(name="RuntimeError", value="boom", timestamp=1)
    assert _exit_code(_execution(exit_code=None, error=err)) == 1


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (SandboxApiException("nope", status_code=404), FILE_NOT_FOUND),
        (SandboxApiException("no such file", status_code=500), FILE_NOT_FOUND),
        (SandboxApiException("denied", status_code=403), PERMISSION_DENIED),
        (SandboxApiException("permission denied", status_code=500), PERMISSION_DENIED),
        (SandboxException("some other error"), None),
    ],
)
def test_classify(exc, expected):
    assert _classify(exc) == expected


def test_execute_returns_combined_output_and_exit_code(backend, sandbox):
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_message("hello", 1)], stderr=[_message("warn", 2)], exit_code=0
    )
    res = backend.execute("echo hello")
    assert res.output == "hello\nwarn"
    assert res.exit_code == 0
    assert res.truncated is False


def test_execute_swallows_sdk_exception(backend, sandbox):
    sandbox.commands.raise_exc = SandboxApiException("connection refused", status_code=502)
    res = backend.execute("ls")
    assert res.exit_code == 1
    assert "connection refused" in res.output


def test_execute_sets_timeout_opts(backend, sandbox):
    backend.execute("sleep 1", timeout=5)
    assert sandbox.commands.last_opts is not None
    assert sandbox.commands.last_opts.timeout.total_seconds() == 5


def test_execute_no_timeout_leaves_opts_unset(backend, sandbox):
    backend.execute("echo hi")
    assert sandbox.commands.last_opts is None


def test_execute_zero_timeout_means_no_deadline(backend, sandbox):
    backend.execute("echo hi", timeout=0)
    assert sandbox.commands.last_opts is None


def test_default_timeout_applied(sandbox):
    backend = OpenSandboxBackend(sandbox, default_timeout=30)  # type: ignore[arg-type]
    backend.execute("echo hi")
    assert sandbox.commands.last_opts is not None
    assert sandbox.commands.last_opts.timeout.total_seconds() == 30


def test_upload_files_writes_bytes(backend, sandbox):
    responses = backend.upload_files([("/workspace/a.txt", b"hello")])
    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].path == "/workspace/a.txt"
    assert sandbox.files_store["/workspace/a.txt"] == b"hello"


def test_upload_files_partial_success(backend, sandbox):
    original = sandbox.files.write_file

    def flaky(path, data, **kw):
        if path == "/bad":
            raise SandboxApiException("permission denied", status_code=403)
        return original(path, data, **kw)

    sandbox.files.write_file = flaky  # type: ignore[assignment]
    responses = backend.upload_files([("/ok.txt", b"x"), ("/bad", b"y")])
    assert responses[0].error is None
    assert responses[1].error == PERMISSION_DENIED
    assert sandbox.files_store["/ok.txt"] == b"x"


def test_upload_retries_after_mkdir_when_parent_missing(backend, sandbox):
    sandbox.files.strict_parents = True
    responses = backend.upload_files([("/workspace/deep/nested/f.txt", b"data")])
    assert responses[0].error is None
    assert sandbox.files_store["/workspace/deep/nested/f.txt"] == b"data"
    assert "/workspace/deep/nested" in sandbox.dirs


def test_download_files_returns_bytes(backend, sandbox):
    sandbox.files_store["/f.bin"] = b"\x00\x01\x02"
    responses = backend.download_files(["/f.bin"])
    assert responses[0].content == b"\x00\x01\x02"
    assert responses[0].error is None


def test_download_missing_maps_to_file_not_found(backend):
    responses = backend.download_files(["/does/not/exist"])
    assert responses[0].content is None
    assert responses[0].error == FILE_NOT_FOUND


def test_read_text_native_no_execute(backend, sandbox):
    sandbox.files_store["/w/notes.txt"] = b"alpha\nbeta\ngamma\n"
    result = backend.read("/w/notes.txt")
    assert result.error is None
    assert result.file_data["content"] == "alpha\nbeta\ngamma"
    assert result.file_data["encoding"] == "utf-8"
    assert sandbox.commands.last_command is None


def test_read_pagination_offset_limit(backend, sandbox):
    sandbox.files_store["/w/n.txt"] = b"l1\nl2\nl3\nl4\n"
    result = backend.read("/w/n.txt", offset=1, limit=2)
    assert result.file_data["content"] == "l2\nl3"


def test_read_offset_beyond_eof_is_error(backend, sandbox):
    sandbox.files_store["/w/n.txt"] = b"only\n"
    result = backend.read("/w/n.txt", offset=5)
    assert result.error == "File '/w/n.txt': Line offset 5 exceeds file length (1 lines)"


def test_read_empty_file_returns_reminder(backend, sandbox):
    sandbox.files_store["/w/empty.txt"] = b""
    result = backend.read("/w/empty.txt")
    assert result.error is None
    assert "empty contents" in result.file_data["content"]


def test_read_missing_maps_to_file_not_found(backend):
    result = backend.read("/does/not/exist.txt")
    assert result.error == f"File '/does/not/exist.txt': {FILE_NOT_FOUND}"


def test_read_normalizes_crlf(backend, sandbox):
    sandbox.files_store["/w/dos.txt"] = b"a\r\nb\r\n"
    result = backend.read("/w/dos.txt")
    assert result.file_data["content"] == "a\nb"


def test_read_binary_extension_returns_base64(backend, sandbox):
    raw = b"\x89PNG\r\n\x1a\n\x00\x01\x02"
    sandbox.files_store["/w/img.png"] = raw
    result = backend.read("/w/img.png")
    assert result.error is None
    assert result.file_data["encoding"] == "base64"
    assert base64.b64decode(result.file_data["content"]) == raw


def test_read_invalid_utf8_text_falls_back_to_base64(backend, sandbox):
    raw = b"\xff\xfe broken"
    sandbox.files_store["/w/data.txt"] = raw
    result = backend.read("/w/data.txt")
    assert result.error is None
    assert result.file_data["encoding"] == "base64"
    assert base64.b64decode(result.file_data["content"]) == raw


def test_read_binary_over_cap_is_error(backend, sandbox):
    sandbox.files_store["/w/big.png"] = b"\x00" * (MAX_BINARY_BYTES + 1)
    result = backend.read("/w/big.png")
    assert result.error is not None
    assert "exceeds maximum preview size" in result.error


def test_read_truncates_huge_text_page(backend, sandbox):
    sandbox.files_store["/w/huge.txt"] = ("\n".join(["x" * 1000] * 600)).encode()
    result = backend.read("/w/huge.txt")
    assert result.error is None
    assert result.file_data["content"].endswith(TRUNCATION_MSG)
    assert len(result.file_data["content"].encode()) <= MAX_OUTPUT_BYTES


async def test_aread_uses_native_path(backend, sandbox):
    sandbox.files_store["/w/a.txt"] = b"hello\n"
    result = await backend.aread("/w/a.txt")
    assert result.error is None
    assert result.file_data["content"] == "hello"
    assert sandbox.commands.last_command is None


def test_id_is_sandbox_id(backend, sandbox):
    assert backend.id == sandbox.id


def test_close_owned_kills_and_closes(sandbox):
    backend = OpenSandboxBackend(sandbox, owns_sandbox=True)  # type: ignore[arg-type]
    backend.close()
    assert sandbox.killed is True
    assert sandbox.closed is True


def test_close_unowned_is_noop(sandbox):
    backend = OpenSandboxBackend(sandbox, owns_sandbox=False)  # type: ignore[arg-type]
    backend.close()
    assert sandbox.killed is False
    assert sandbox.closed is False


def test_context_manager_closes_owned(sandbox):
    with OpenSandboxBackend(sandbox, owns_sandbox=True) as backend:  # type: ignore[arg-type]
        assert backend.id == sandbox.id
    assert sandbox.killed is True


@pytest.mark.skipif(
    os.environ.get("RUN_OPENSANDBOX_INTEGRATION") != "1",
    reason="需设置 RUN_OPENSANDBOX_INTEGRATION=1 并有运行中的 OpenSandbox 服务端",
)
def test_integration_full_pipeline():
    with OpenSandboxBackend.create(image="python:3.11") as backend:
        assert backend.execute("echo hi").output == "hi"
        assert backend.write("/workspace/t.txt", "alpha\nbeta\n").error is None
        read = backend.read("/workspace/t.txt")
        assert read.error is None
        assert "alpha" in read.file_data["content"]
        grep = backend.grep("beta", "/workspace")
        assert any(m["line"] == 2 for m in (grep.matches or []))
