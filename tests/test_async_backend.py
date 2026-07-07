"""AsyncOpenSandboxBackend 的单元测试(原生异步)。

使用一个异步内存假沙箱,验证三个原生异步原语、辅助逻辑、生命周期,以及“派生的
异步文件操作(如 ``aread``)确实走原生 ``aexecute``”这一关键点。无需服务端,跨平台。
pytest-asyncio 以 asyncio_mode=auto 运行 ``async def`` 测试。
"""

from __future__ import annotations

import posixpath
import shlex

import pytest

from deepagents.backends.protocol import FILE_NOT_FOUND, PERMISSION_DENIED
from deepagents_opensandbox import AsyncOpenSandboxBackend
from opensandbox.exceptions import SandboxApiException
from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage


# --------------------------------------------------------------------------- #
# 异步假对象(stub)
# --------------------------------------------------------------------------- #
def _msg(text: str, ts: int, *, is_error: bool = False) -> OutputMessage:
    return OutputMessage(text=text, timestamp=ts, is_error=is_error)


def _execution(*, stdout=(), stderr=(), exit_code=0, error=None) -> Execution:
    return Execution(
        logs=ExecutionLogs(stdout=list(stdout), stderr=list(stderr)),
        exit_code=exit_code,
        error=error,
    )


class AsyncFakeCommands:
    def __init__(self, sandbox: "AsyncFakeSandbox") -> None:
        self._sandbox = sandbox
        self.responder = None  # callable(command, opts) -> Execution
        self.raise_exc: Exception | None = None
        self.last_command: str | None = None
        self.last_opts = None

    async def run(self, command, *, opts=None, handlers=None) -> Execution:
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
        return _execution(stdout=[_msg("ok", 1)], exit_code=0)


class AsyncFakeFiles:
    def __init__(self, sandbox: "AsyncFakeSandbox") -> None:
        self._sandbox = sandbox
        self.strict_parents = False

    async def write_file(self, path, data, *, encoding="utf-8", mode=755, owner=None, group=None) -> None:
        parent = posixpath.dirname(path)
        if self.strict_parents and parent not in ("", "/") and parent not in self._sandbox.dirs:
            raise SandboxApiException("No such file or directory", status_code=500)
        self._sandbox.files_store[path] = data if isinstance(data, bytes) else str(data).encode(encoding)

    async def read_bytes(self, path, *, range_header=None, offset=None, limit=None) -> bytes:
        if path not in self._sandbox.files_store:
            raise SandboxApiException(f"file not found: {path}", status_code=404)
        return self._sandbox.files_store[path]


class AsyncFakeSandbox:
    """opensandbox.Sandbox(异步)的最小替身。"""

    def __init__(self, sandbox_id: str = "sbx-async-1") -> None:
        self.id = sandbox_id
        self.files_store: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.commands = AsyncFakeCommands(self)
        self.files = AsyncFakeFiles(self)
        self.killed = False
        self.closed = False

    async def kill(self) -> None:
        self.killed = True

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def sandbox() -> AsyncFakeSandbox:
    return AsyncFakeSandbox()


@pytest.fixture
def backend(sandbox: AsyncFakeSandbox) -> AsyncOpenSandboxBackend:
    return AsyncOpenSandboxBackend(sandbox)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# aexecute
# --------------------------------------------------------------------------- #
async def test_aexecute_returns_combined_output_and_exit_code(backend, sandbox):
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_msg("hello", 1)], stderr=[_msg("warn", 2)], exit_code=0
    )
    res = await backend.aexecute("echo hello")
    assert res.output == "hello\nwarn"
    assert res.exit_code == 0


async def test_aexecute_empty_command_is_error(backend):
    res = await backend.aexecute("")
    assert res.exit_code == 1
    assert "non-empty" in res.output


async def test_aexecute_swallows_sdk_exception(backend, sandbox):
    sandbox.commands.raise_exc = SandboxApiException("connection refused", status_code=502)
    res = await backend.aexecute("ls")
    assert res.exit_code == 1
    assert "connection refused" in res.output


async def test_aexecute_sets_timeout_opts(backend, sandbox):
    await backend.aexecute("sleep 1", timeout=5)
    assert sandbox.commands.last_opts is not None
    assert sandbox.commands.last_opts.timeout.total_seconds() == 5


async def test_default_timeout_applied(sandbox):
    backend = AsyncOpenSandboxBackend(sandbox, default_timeout=30)  # type: ignore[arg-type]
    await backend.aexecute("echo hi")
    assert sandbox.commands.last_opts is not None
    assert sandbox.commands.last_opts.timeout.total_seconds() == 30


# --------------------------------------------------------------------------- #
# aupload_files / adownload_files
# --------------------------------------------------------------------------- #
async def test_aupload_files_writes_bytes(backend, sandbox):
    responses = await backend.aupload_files([("/workspace/a.txt", b"hello")])
    assert responses[0].error is None
    assert sandbox.files_store["/workspace/a.txt"] == b"hello"


async def test_aupload_files_partial_success(backend, sandbox):
    original = sandbox.files.write_file

    async def flaky(path, data, **kw):
        if path == "/bad":
            raise SandboxApiException("permission denied", status_code=403)
        return await original(path, data, **kw)

    sandbox.files.write_file = flaky  # type: ignore[assignment]
    responses = await backend.aupload_files([("/ok.txt", b"x"), ("/bad", b"y")])
    assert responses[0].error is None
    assert responses[1].error == PERMISSION_DENIED


async def test_aupload_retries_after_mkdir_when_parent_missing(backend, sandbox):
    sandbox.files.strict_parents = True
    responses = await backend.aupload_files([("/workspace/deep/nested/f.txt", b"data")])
    assert responses[0].error is None
    assert sandbox.files_store["/workspace/deep/nested/f.txt"] == b"data"
    assert "/workspace/deep/nested" in sandbox.dirs


async def test_adownload_files_returns_bytes(backend, sandbox):
    sandbox.files_store["/f.bin"] = b"\x00\x01\x02"
    responses = await backend.adownload_files(["/f.bin"])
    assert responses[0].content == b"\x00\x01\x02"
    assert responses[0].error is None


async def test_adownload_missing_maps_to_file_not_found(backend):
    responses = await backend.adownload_files(["/nope"])
    assert responses[0].content is None
    assert responses[0].error == FILE_NOT_FOUND


# --------------------------------------------------------------------------- #
# 派生异步文件操作确实走原生 aexecute
# --------------------------------------------------------------------------- #
async def test_aread_is_derived_from_native_aexecute(backend, sandbox):
    # aread 会通过 aexecute 运行服务端读取脚本;这里让 aexecute 返回该脚本约定的
    # JSON,以证明 BaseSandbox 的派生异步操作确实走我们重写的原生 aexecute。
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_msg('{"encoding": "utf-8", "content": "hello world"}', 1)],
        exit_code=0,
    )
    result = await backend.aread("/workspace/x.txt")
    assert result.error is None
    assert result.file_data["content"] == "hello world"


# --------------------------------------------------------------------------- #
# 仅限异步:同步原语必须拒绝
# --------------------------------------------------------------------------- #
def test_sync_primitives_raise(backend):
    with pytest.raises(NotImplementedError):
        backend.execute("ls")
    with pytest.raises(NotImplementedError):
        backend.upload_files([("/a", b"x")])
    with pytest.raises(NotImplementedError):
        backend.download_files(["/a"])


# --------------------------------------------------------------------------- #
# 标识与生命周期
# --------------------------------------------------------------------------- #
async def test_id_is_sandbox_id(backend, sandbox):
    assert backend.id == sandbox.id


async def test_aclose_owned_kills_and_closes(sandbox):
    backend = AsyncOpenSandboxBackend(sandbox, owns_sandbox=True)  # type: ignore[arg-type]
    await backend.aclose()
    assert sandbox.killed is True
    assert sandbox.closed is True


async def test_aclose_unowned_is_noop(sandbox):
    backend = AsyncOpenSandboxBackend(sandbox, owns_sandbox=False)  # type: ignore[arg-type]
    await backend.aclose()
    assert sandbox.killed is False
    assert sandbox.closed is False


async def test_async_context_manager_closes_owned(sandbox):
    async with AsyncOpenSandboxBackend(sandbox, owns_sandbox=True) as backend:  # type: ignore[arg-type]
        assert backend.id == sandbox.id
    assert sandbox.killed is True
