"""面向高并发的**原生异步** deepagents 后端。

:class:`AsyncOpenSandboxBackend` 封装 OpenSandbox 的**异步** SDK
(``opensandbox.Sandbox``,基于异步 httpx),原生实现三个异步原语
``aexecute``/``aupload_files``/``adownload_files``。由于 ``BaseSandbox`` 的异步文件
操作(``als``/``aread``/``awrite``/``aedit``/``aglob``/``agrep``)内部都调用
``self.aexecute``/``self.aupload_files``,因此**整个异步接口都会走原生协程**,
无需 ``asyncio.to_thread`` 线程卸载。

为何要它
--------
:class:`~deepagents_opensandbox.backend.OpenSandboxBackend`(同步)配异步智能体时,
deepagents 会通过 ``asyncio.to_thread`` 把阻塞调用丢进线程池——单智能体够用,但在
**高并发 / 批量评测 / RL rollout** 下,并发受线程池上限约束、长时 SSE 流会占满线程、
超时也无法真正中断阻塞的 socket 读。原生异步用协程承载在途 I/O,可轻松扩展到数千
并发,并支持干净的取消与超时。

仅限异步
--------
本后端只服务异步智能体(``agent.ainvoke(...)``)。同步原语
``execute``/``upload_files``/``download_files`` 会抛出 ``NotImplementedError``——
同步场景请改用 :class:`~deepagents_opensandbox.backend.OpenSandboxBackend`。

Example:
    ```python
    from deepagents import create_deep_agent
    from deepagents_opensandbox import AsyncOpenSandboxBackend

    async with await AsyncOpenSandboxBackend.create(image="python:3.11") as backend:
        agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
    ```
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
    DEFAULT_SANDBOX_TIMEOUT,
    _classify_error,
    _combine_output,
    _resolve_exit_code,
)

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注的导入
    from opensandbox.config import ConnectionConfig
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["AsyncOpenSandboxBackend"]

# 同步原语被调用时给出的提示(功能性字符串,保持英文)。
_SYNC_UNSUPPORTED = (
    "AsyncOpenSandboxBackend is async-only; use it with async agents "
    "(agent.ainvoke(...)). For synchronous agents use OpenSandboxBackend."
)


class AsyncOpenSandboxBackend(BaseSandbox):
    """由 OpenSandbox 异步沙箱支撑的原生异步 deepagents 后端。

    Args:
        sandbox: 一个就绪的异步 ``Sandbox`` 实例。
        owns_sandbox: 若为 ``True``,:meth:`aclose` 会 ``kill()`` 远端沙箱并释放
            本地资源;由 :meth:`create` 自动置为 ``True``。
        default_timeout: 调用 ``aexecute`` 未显式指定 ``timeout`` 时使用的默认
            单命令超时(秒)。``None`` 表示客户端不施加超时。
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

    # -- 构造辅助方法(异步,因为 Sandbox.create/connect 是协程)-------------

    @classmethod
    async def create(
        cls,
        image: SandboxImageSpec | str = DEFAULT_IMAGE,
        *,
        connection_config: ConnectionConfig | None = None,
        timeout: timedelta | None = DEFAULT_SANDBOX_TIMEOUT,
        default_timeout: int | None = None,
        **create_kwargs: object,
    ) -> AsyncOpenSandboxBackend:
        """新建一个异步 OpenSandbox 沙箱并封装它(后端拥有其生命周期)。"""
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
        """按 id 连接到已运行的异步沙箱并封装它(不拥有其生命周期)。"""
        sandbox = await Sandbox.connect(
            sandbox_id,
            connection_config=connection_config,
            **connect_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=False, default_timeout=default_timeout)

    # -- 自省 ----------------------------------------------------------------

    @property
    def sandbox(self) -> Sandbox:
        """底层的 OpenSandbox 异步 ``Sandbox`` 实例。"""
        return self._sandbox

    @property
    def id(self) -> str:
        """后端的稳定标识(即 OpenSandbox 沙箱 id)。"""
        return self._sandbox.id

    # -- 原生异步原语 --------------------------------------------------------

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """在沙箱内异步执行 ``command``,返回合并输出与退出码。

        这是 ``BaseSandbox`` 派生全部异步文件操作的核心原语。语义与同步后端一致
        (见 :class:`~deepagents_opensandbox.backend.OpenSandboxBackend`),但全程
        ``await``,不占用线程池。
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        opts: RunCommandOpts | None = None
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
            exit_code=_resolve_exit_code(execution),
            truncated=False,
        )

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """把若干 ``(path, bytes)`` 异步写入沙箱,支持部分成功。"""
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
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_classify_error(exc) or str(exc),
                    )
                )
        return responses

    # -- 同步原语:不支持(本后端仅限异步)----------------------------------
    # BaseSandbox 把这三个方法声明为抽象方法,必须实现;此处显式拒绝,
    # 引导用户使用异步接口或改用同步后端。

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError(_SYNC_UNSUPPORTED)

    # -- 生命周期 ------------------------------------------------------------

    async def aclose(self) -> None:
        """异步释放资源:拥有沙箱时终止远端并关闭本地传输,否则为空操作。"""
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
        return (
            f"AsyncOpenSandboxBackend(id={self._sandbox.id!r}, "
            f"owns_sandbox={self._owns_sandbox})"
        )
