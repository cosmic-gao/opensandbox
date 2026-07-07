"""OpenSandbox 支撑的 deepagents 执行后端(同步)。

只实现 execute/upload_files/download_files 三个原语,ls/read/write/edit/glob/grep
由 deepagents 的 BaseSandbox 派生。设计与用法见 README。
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from datetime import timedelta
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox import SandboxSync
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import Execution, RunCommandOpts

if TYPE_CHECKING:  # pragma: no cover
    from opensandbox.config import ConnectionConfigSync
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["OpenSandboxBackend"]

DEFAULT_IMAGE = "python:3.11"
DEFAULT_LIFETIME = timedelta(minutes=30)


class OpenSandboxBackend(BaseSandbox):
    """由 OpenSandbox 沙箱支撑的 deepagents 后端。

    Args:
        sandbox: 就绪的 ``SandboxSync`` 实例。
        owns_sandbox: 为 ``True`` 时 :meth:`close` 会终止远端沙箱;:meth:`create`
            会置为 ``True``。封装你自己管理的沙箱时保持 ``False``。
        default_timeout: ``execute`` 未指定 ``timeout`` 时的默认单命令超时(秒);
            ``None`` 表示不限。

    Example:
        ```python
        with OpenSandboxBackend.create(image="python:3.11") as backend:
            agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
        ```
    """

    def __init__(
        self,
        sandbox: SandboxSync,
        *,
        owns_sandbox: bool = False,
        default_timeout: int | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._owns_sandbox = owns_sandbox
        self._default_timeout = default_timeout

    @classmethod
    def create(
        cls,
        image: SandboxImageSpec | str = DEFAULT_IMAGE,
        *,
        connection_config: ConnectionConfigSync | None = None,
        timeout: timedelta | None = DEFAULT_LIFETIME,
        default_timeout: int | None = None,
        **create_kwargs: object,
    ) -> OpenSandboxBackend:
        """新建并拥有一个沙箱(:meth:`close` 会终止它)。

        ``connection_config`` 省略时 SDK 读环境变量 ``OPEN_SANDBOX_DOMAIN`` /
        ``OPEN_SANDBOX_API_KEY``;额外 kwargs 透传给 ``SandboxSync.create``
        (``env`` / ``resource`` / ``metadata`` / ``network_policy`` ...)。
        """
        sandbox = SandboxSync.create(
            image,
            connection_config=connection_config,
            timeout=timeout,
            **create_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=True, default_timeout=default_timeout)

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        *,
        connection_config: ConnectionConfigSync | None = None,
        default_timeout: int | None = None,
        **connect_kwargs: object,
    ) -> OpenSandboxBackend:
        """连接到已运行的沙箱并封装;不拥有它,:meth:`close` 不会终止它。"""
        sandbox = SandboxSync.connect(
            sandbox_id,
            connection_config=connection_config,
            **connect_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=False, default_timeout=default_timeout)

    @property
    def sandbox(self) -> SandboxSync:
        return self._sandbox

    @property
    def id(self) -> str:
        return self._sandbox.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """执行命令,返回合并输出与退出码;绝不抛异常(错误转成非零退出码)。

        这是 BaseSandbox 派生全部文件操作的唯一原语,必须如实反映退出码——
        ``write`` 靠非零退出码判断“文件已存在”。
        """
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
            execution = self._sandbox.commands.run(command, opts=opts)
        except Exception as exc:  # noqa: BLE001 - 后端只返回错误,不抛出
            logger.debug("execute failed for command %r", command, exc_info=exc)
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

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """写入若干 ``(path, bytes)``,支持部分成功;父目录缺失时 ``mkdir -p`` 后重试一次。"""
        return [self._upload_one(path, data) for path, data in files]

    def _upload_one(self, path: str, data: bytes) -> FileUploadResponse:
        try:
            self._sandbox.files.write_file(path, data)
            return FileUploadResponse(path=path, error=None)
        except Exception as exc:  # noqa: BLE001 - 逐个文件上报,绝不抛出
            parent = posixpath.dirname(path)
            if parent and parent not in ("", "/"):
                mkdir = self.execute(f"mkdir -p {shlex.quote(parent)}")
                if mkdir.exit_code == 0:
                    try:
                        self._sandbox.files.write_file(path, data)
                        return FileUploadResponse(path=path, error=None)
                    except Exception as retry_exc:  # noqa: BLE001
                        exc = retry_exc
            logger.debug("upload failed for %s", path, exc_info=exc)
            return FileUploadResponse(path=path, error=_classify_error(exc) or str(exc))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """读取若干路径的原始字节,支持部分成功。"""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:  # noqa: BLE001 - 逐个文件上报,绝不抛出
                logger.debug("download failed for %s", path, exc_info=exc)
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=_classify_error(exc) or str(exc))
                )
        return responses

    def close(self) -> None:
        """终止并释放资源;仅当拥有沙箱(``owns_sandbox=True``)时生效,否则为空操作。"""
        if not self._owns_sandbox:
            return
        try:
            self._sandbox.kill()
        except SandboxException as exc:
            logger.warning("Failed to kill sandbox %s: %s", self._sandbox.id, exc)
        finally:
            self._sandbox.close()

    def __enter__(self) -> OpenSandboxBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"OpenSandboxBackend(id={self._sandbox.id!r}, owns_sandbox={self._owns_sandbox})"


def _combine_output(execution: Execution) -> str:
    """按时间戳合并 stdout 与 stderr;无输出时返回空串。

    不能用 ``"<no output>"`` 之类哨兵替代空串——BaseSandbox 解析器把空输出视为
    “无结果”(如 grep 无匹配)。
    """
    messages = [*execution.logs.stdout, *execution.logs.stderr]
    messages.sort(key=lambda m: m.timestamp)
    return "\n".join(msg.text.rstrip("\n") for msg in messages)


def _exit_code(execution: Execution) -> int:
    """返回退出码;服务端缺省时按是否有 error 推断,保证 ``write`` 前置检查可靠。"""
    if execution.exit_code is not None:
        return execution.exit_code
    return 1 if execution.error is not None else 0


def _classify_error(exc: Exception) -> str | None:
    """把 SDK 异常映射为 deepagents 文件错误码,无法识别时返回 ``None``。"""
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status == 404 or "not found" in message or "no such file" in message:
        return FILE_NOT_FOUND
    if status == 403 or "permission denied" in message:
        return PERMISSION_DENIED
    return None
