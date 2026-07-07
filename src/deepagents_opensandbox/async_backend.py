"""OpenSandbox 支撑的**原生异步** deepagents 后端,用于高并发 / 评测 / RL。

封装异步 ``opensandbox.Sandbox``,原生实现 aexecute/aupload_files/adownload_files;
BaseSandbox 的异步文件操作都转发到这三者,故整个异步接口都是原生协程(不走线程池)。
仅供异步智能体(``agent.ainvoke``)——同步原语会抛 ``NotImplementedError``。详见 README。
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from datetime import timedelta
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox import Sandbox
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import RunCommandOpts

from deepagents_opensandbox.backend import (
    DEFAULT_IMAGE,
    DEFAULT_LIFETIME,
    _classify_error,
    _combine_output,
    _exit_code,
)

if TYPE_CHECKING:  # pragma: no cover
    from opensandbox.config import ConnectionConfig
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["AsyncOpenSandboxBackend"]

_SYNC_UNSUPPORTED = (
    "AsyncOpenSandboxBackend is async-only; use it with async agents "
    "(agent.ainvoke(...)). For synchronous agents use OpenSandboxBackend."
)


class AsyncOpenSandboxBackend(BaseSandbox):
    """由 OpenSandbox 异步沙箱支撑的原生异步 deepagents 后端。

    Args:
        sandbox: 就绪的异步 ``Sandbox`` 实例。
        owns_sandbox: 为 ``True`` 时 :meth:`aclose` 会终止远端沙箱;:meth:`create`
            会置为 ``True``。
        default_timeout: ``aexecute`` 未指定 ``timeout`` 时的默认单命令超时(秒);
            ``None`` 表示不限。
    """

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
        """新建并拥有一个异步沙箱;额外 kwargs 透传给 ``Sandbox.create``。"""
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
        """连接到已运行的异步沙箱并封装;不拥有它,:meth:`aclose` 不会终止它。"""
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
        """异步执行命令,返回合并输出与退出码;语义同 ``OpenSandboxBackend.execute``,但全程 ``await``。"""
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        opts: RunCommandOpts | None = None
        # timeout<=0 表示不设客户端超时,交给服务端策略,而非施加 0 秒截止时间。
        if effective_timeout is not None and effective_timeout > 0:
            opts = RunCommandOpts(timeout=timedelta(seconds=effective_timeout))

        try:
            execution = await self._sandbox.commands.run(command, opts=opts)
        except Exception as exc:  # noqa: BLE001 - 后端只返回错误,不抛出
            logger.debug("aexecute failed for command %r", command, exc_info=exc)
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=_combine_output(execution),
            exit_code=_exit_code(execution),
            truncated=False,
        )

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """异步写入若干 ``(path, bytes)``,支持部分成功;父目录缺失时 ``mkdir -p`` 后重试一次。"""
        return [await self._aupload_one(path, data) for path, data in files]

    async def _aupload_one(self, path: str, data: bytes) -> FileUploadResponse:
        try:
            await self._sandbox.files.write_file(path, data)
            return FileUploadResponse(path=path, error=None)
        except Exception as exc:  # noqa: BLE001 - 逐个文件上报,绝不抛出
            parent = posixpath.dirname(path)
            if parent and parent not in ("", "/"):
                mkdir = await self.aexecute(f"mkdir -p {shlex.quote(parent)}")
                if mkdir.exit_code == 0:
                    try:
                        await self._sandbox.files.write_file(path, data)
                        return FileUploadResponse(path=path, error=None)
                    except Exception as retry_exc:  # noqa: BLE001
                        exc = retry_exc
            logger.debug("aupload failed for %s", path, exc_info=exc)
            return FileUploadResponse(path=path, error=_classify_error(exc) or str(exc))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """逐个从沙箱异步读取原始字节,支持部分成功。"""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = await self._sandbox.files.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:  # noqa: BLE001 - 逐个文件上报,绝不抛出
                logger.debug("adownload failed for %s", path, exc_info=exc)
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=_classify_error(exc) or str(exc))
                )
        return responses

    # BaseSandbox 要求实现这三个抽象方法;本后端仅限异步,故显式拒绝。
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    async def aclose(self) -> None:
        """终止并释放资源;仅当拥有沙箱(``owns_sandbox=True``)时生效,否则为空操作。"""
        if not self._owns_sandbox:
            return
        try:
            await self._sandbox.kill()
        except SandboxException as exc:
            logger.warning("Failed to kill sandbox %s: %s", self._sandbox.id, exc)
        finally:
            await self._sandbox.close()

    async def __aenter__(self) -> AsyncOpenSandboxBackend:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncOpenSandboxBackend(id={self._sandbox.id!r}, owns_sandbox={self._owns_sandbox})"
