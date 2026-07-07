"""OpenSandboxBackend 的单元测试。

测试使用一个内存版假沙箱,只模拟后端实际用到的那一小片 OpenSandbox
``SandboxSync`` 接口(``id``、``commands.run``、``files.write_file``/``read_bytes``、
``kill``/``close``)。它们无需运行中的沙箱服务端,且跨平台:因为测试直接验证三个
原语与辅助函数,而非 ``BaseSandbox`` 叠在 ``execute`` 之上的那些服务端脚本。

一个可选的集成测试(``RUN_OPENSANDBOX_INTEGRATION=1``)会针对真实服务端验证
完整的派生链路。
"""

from __future__ import annotations

import os
import posixpath
import shlex

import pytest

from deepagents.backends.protocol import FILE_NOT_FOUND, PERMISSION_DENIED
from deepagents_opensandbox import OpenSandboxBackend
from deepagents_opensandbox.backend import (
    _classify_error,
    _combine_output,
    _exit_code,
)
from opensandbox.exceptions import SandboxApiException, SandboxException
from opensandbox.models.execd import Execution, ExecutionError, ExecutionLogs, OutputMessage


# --------------------------------------------------------------------------- #
# 假对象(stub)
# --------------------------------------------------------------------------- #
def _message(text: str, ts: int, *, is_error: bool = False) -> OutputMessage:
    return OutputMessage(text=text, timestamp=ts, is_error=is_error)


def _execution(*, stdout=(), stderr=(), exit_code=0, error=None) -> Execution:
    return Execution(
        logs=ExecutionLogs(stdout=list(stdout), stderr=list(stderr)),
        exit_code=exit_code,
        error=error,
    )


class FakeCommands:
    """模拟 ``sandbox.commands``。"""

    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox
        self.responder = None  # callable(command, opts) -> Execution
        self.raise_exc: Exception | None = None
        self.last_command: str | None = None
        self.last_opts = None

    def run(self, command, *, opts=None, handlers=None) -> Execution:
        self.last_command = command
        self.last_opts = opts
        if self.raise_exc is not None:
            raise self.raise_exc
        # 模拟 `mkdir -p <dir>`,让上传重试路径能够成功。
        if command.startswith("mkdir -p "):
            target = shlex.split(command[len("mkdir -p "):])[0]
            self._sandbox.dirs.add(target)
            return _execution(exit_code=0)
        if self.responder is not None:
            return self.responder(command, opts)
        return _execution(stdout=[_message("ok", 1)], exit_code=0)


class FakeFiles:
    """模拟 ``sandbox.files``。"""

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
    """opensandbox.SandboxSync 的最小替身。"""

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


# --------------------------------------------------------------------------- #
# 辅助函数单测
# --------------------------------------------------------------------------- #
def test_combine_output_orders_by_timestamp():
    execution = _execution(
        stdout=[_message("first", 1), _message("third", 3)],
        stderr=[_message("second", 2, is_error=True)],
    )
    assert _combine_output(execution) == "first\nsecond\nthird"


def test_combine_output_empty_is_empty_string():
    # 关键:BaseSandbox 解析器把空输出视为“无结果”(如 grep 无匹配),
    # 绝不能替换成哨兵字符串。
    assert _combine_output(_execution()) == ""


def test_combine_output_strips_trailing_newlines_per_message():
    execution = _execution(stdout=[_message("line\n", 1), _message("next\n", 2)])
    assert _combine_output(execution) == "line\nnext"


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
def test_classify_error(exc, expected):
    assert _classify_error(exc) == expected


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #
def test_execute_returns_combined_output_and_exit_code(backend, sandbox):
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_message("hello", 1)], stderr=[_message("warn", 2)], exit_code=0
    )
    res = backend.execute("echo hello")
    assert res.output == "hello\nwarn"
    assert res.exit_code == 0
    assert res.truncated is False


def test_execute_empty_command_is_error(backend):
    res = backend.execute("")
    assert res.exit_code == 1
    assert "non-empty" in res.output


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


# --------------------------------------------------------------------------- #
# upload_files / download_files
# --------------------------------------------------------------------------- #
def test_upload_files_writes_bytes(backend, sandbox):
    responses = backend.upload_files([("/workspace/a.txt", b"hello")])
    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].path == "/workspace/a.txt"
    assert sandbox.files_store["/workspace/a.txt"] == b"hello"


def test_upload_files_partial_success(backend, sandbox):
    # 让某个特定路径即使重试后仍失败,以验证部分成功语义。
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
    # 父目录不存在 -> 首次写入失败 -> 后端 mkdir -p -> 重试成功。
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


# --------------------------------------------------------------------------- #
# 标识与生命周期
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 可选集成测试(真实服务端 + 真实派生文件操作)
# --------------------------------------------------------------------------- #
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
