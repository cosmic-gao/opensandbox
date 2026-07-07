"""OpenSandbox 原生异步 deepagents 后端,面向高并发;仅限异步智能体(``ainvoke``)。"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import shlex
from datetime import timedelta
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox import Sandbox
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import RunCommandOpts

from deepagents_opensandbox.backend import (
    DEFAULT_IMAGE,
    DEFAULT_LIFETIME,
    _classify,
    _exit_code,
    _output,
    _read,
    _read_error,
)

if TYPE_CHECKING:
    from opensandbox.config import ConnectionConfig
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["AsyncOpenSandboxBackend"]

_SYNC_UNSUPPORTED = (
    "AsyncOpenSandboxBackend is async-only; use it with async agents "
    "(agent.ainvoke(...)). For synchronous agents use OpenSandboxBackend."
)


class AsyncOpenSandboxBackend(BaseSandbox):
    """封装异步 ``Sandbox`` 的 deepagents 沙箱后端(async-only)。"""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        owns_sandbox: bool = False,
        default_timeout: int | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._owns_sandbox = owns_sandbox
        self._default_timeout = default_timeout

    @classmethod
    async def create(
        cls,
        image: SandboxImageSpec | str = DEFAULT_IMAGE,
        *,
        connection_config: ConnectionConfig | None = None,
        timeout: timedelta | None = DEFAULT_LIFETIME,
        default_timeout: int | None = None,
        **create_kwargs: object,
    ) -> AsyncOpenSandboxBackend:
        """新建并拥有一个沙箱;省略 ``connection_config`` 时 SDK 读环境变量。"""
        sandbox = await Sandbox.create(
            image,
            connection_config=connection_config,
            timeout=timeout,
            **create_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=True, default_timeout=default_timeout)

    @classmethod
    async def connect(
        cls,
        sandbox_id: str,
        *,
        connection_config: ConnectionConfig | None = None,
        default_timeout: int | None = None,
        **connect_kwargs: object,
    ) -> AsyncOpenSandboxBackend:
        """连接已运行的沙箱;不拥有它,:meth:`aclose` 不会终止它。"""
        sandbox = await Sandbox.connect(
            sandbox_id,
            connection_config=connection_config,
            **connect_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=False, default_timeout=default_timeout)

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def id(self) -> str:
        return self._sandbox.id

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """异步执行命令;错误转为非零退出码,绝不抛异常。"""
        timeout = timeout if timeout is not None else self._default_timeout
        opts = (
            RunCommandOpts(timeout=timedelta(seconds=timeout))
            if timeout is not None and timeout > 0
            else None
        )
        try:
            execution = await self._sandbox.commands.run(command, opts=opts)
        except Exception as exc:
            logger.debug("aexecute failed: %r", command, exc_info=exc)
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
            )
        return ExecuteResponse(output=_output(execution), exit_code=_exit_code(execution))

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """经 SDK 文件传输原生异步读取;语义同 ``OpenSandboxBackend.read``。"""
        try:
            raw = await self._sandbox.files.read_bytes(file_path)
        except Exception as exc:
            logger.debug("aread failed: %s", file_path, exc_info=exc)
            return _read_error(file_path, exc)
        return _read(file_path, raw, offset, limit)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """并发写入,支持部分成功,结果保序。"""
        return list(await asyncio.gather(*(self._put(path, data) for path, data in files)))

    async def _put(self, path: str, data: bytes) -> FileUploadResponse:
        try:
            await self._sandbox.files.write_file(path, data)
            return FileUploadResponse(path=path)
        except Exception as exc:
            parent = posixpath.dirname(path)
            if (
                parent not in ("", "/")
                and (await self.aexecute(f"mkdir -p {shlex.quote(parent)}")).exit_code == 0
            ):
                try:
                    await self._sandbox.files.write_file(path, data)
                    return FileUploadResponse(path=path)
                except Exception as retry_exc:
                    exc = retry_exc
            logger.debug("aupload failed: %s", path, exc_info=exc)
            return FileUploadResponse(path=path, error=_classify(exc) or str(exc))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """并发读取,支持部分成功,结果保序。"""
        return list(await asyncio.gather(*(self._get(path) for path in paths)))

    async def _get(self, path: str) -> FileDownloadResponse:
        try:
            return FileDownloadResponse(
                path=path, content=await self._sandbox.files.read_bytes(path)
            )
        except Exception as exc:
            logger.debug("adownload failed: %s", path, exc_info=exc)
            return FileDownloadResponse(path=path, error=_classify(exc) or str(exc))

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    async def aclose(self) -> None:
        """仅当拥有沙箱时终止并释放;否则为空操作。"""
        if not self._owns_sandbox:
            return
        try:
            await self._sandbox.kill()
        except SandboxException as exc:
            logger.warning("kill sandbox %s failed: %s", self._sandbox.id, exc)
        finally:
            await self._sandbox.close()

    async def __aenter__(self) -> AsyncOpenSandboxBackend:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncOpenSandboxBackend(id={self._sandbox.id!r}, owns_sandbox={self._owns_sandbox})"
