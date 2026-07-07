"""AsyncOpenSandboxBackend 单元测试:异步内存假沙箱驱动原生原语、原生 aread 与并发批量。"""

from __future__ import annotations

import asyncio
import posixpath
import shlex

import pytest

from deepagents.backends.protocol import FILE_NOT_FOUND, PERMISSION_DENIED
from deepagents_opensandbox import AsyncOpenSandboxBackend
from opensandbox.exceptions import SandboxApiException
from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage


def _message(text: str, ts: int, *, is_error: bool = False) -> OutputMessage:
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
        self.responder = None
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
        return _execution(stdout=[_message("ok", 1)], exit_code=0)


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


async def test_aexecute_returns_combined_output_and_exit_code(backend, sandbox):
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_message("hello", 1)], stderr=[_message("warn", 2)], exit_code=0
    )
    res = await backend.aexecute("echo hello")
    assert res.output == "hello\nwarn"
    assert res.exit_code == 0


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


async def test_adownload_preserves_order_under_concurrency(backend, sandbox):
    sandbox.files_store["/a"] = b"A"
    sandbox.files_store["/b"] = b"B"
    original = sandbox.files.read_bytes

    async def slow_first(path, **kw):
        if path == "/a":
            await asyncio.sleep(0.05)
        return await original(path, **kw)

    sandbox.files.read_bytes = slow_first  # type: ignore[assignment]
    responses = await backend.adownload_files(["/a", "/b"])
    assert [r.path for r in responses] == ["/a", "/b"]
    assert [r.content for r in responses] == [b"A", b"B"]


async def test_aread_native_no_execute(backend, sandbox):
    sandbox.files_store["/workspace/x.txt"] = b"hello world\n"
    result = await backend.aread("/workspace/x.txt")
    assert result.error is None
    assert result.file_data["content"] == "hello world"
    assert sandbox.commands.last_command is None


async def test_aread_pagination_offset_limit(backend, sandbox):
    sandbox.files_store["/w/n.txt"] = b"l1\nl2\nl3\nl4\n"
    result = await backend.aread("/w/n.txt", offset=1, limit=2)
    assert result.file_data["content"] == "l2\nl3"


async def test_aread_missing_maps_to_file_not_found(backend):
    result = await backend.aread("/nope.txt")
    assert result.error == f"File '/nope.txt': {FILE_NOT_FOUND}"


async def test_als_derives_from_native_aexecute(backend, sandbox):
    sandbox.commands.responder = lambda cmd, opts: _execution(
        stdout=[_message('{"path": "/w/a.txt", "is_dir": false}', 1)],
        exit_code=0,
    )
    result = await backend.als("/w")
    assert result.error is None
    assert result.entries == [{"path": "/w/a.txt", "is_dir": False}]


def test_sync_primitives_raise(backend):
    with pytest.raises(NotImplementedError):
        backend.execute("ls")
    with pytest.raises(NotImplementedError):
        backend.upload_files([("/a", b"x")])
    with pytest.raises(NotImplementedError):
        backend.download_files(["/a"])


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
