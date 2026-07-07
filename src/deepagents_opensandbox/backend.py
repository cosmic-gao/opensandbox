"""OpenSandbox 同步 deepagents 后端:execute/upload/download 原语 + 原生 read。"""

from __future__ import annotations

import asyncio
import base64
import logging
import posixpath
import shlex
from datetime import timedelta
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
)
from deepagents.backends.sandbox import (
    MAX_BINARY_BYTES,
    MAX_OUTPUT_BYTES,
    TRUNCATION_MSG,
    BaseSandbox,
)
from deepagents.backends.utils import EMPTY_CONTENT_WARNING, _get_file_type
from opensandbox import SandboxSync
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import RunCommandOpts

if TYPE_CHECKING:
    from opensandbox.config import ConnectionConfigSync
    from opensandbox.models.execd import Execution
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["OpenSandboxBackend"]

DEFAULT_IMAGE = "python:3.11"
DEFAULT_LIFETIME = timedelta(minutes=30)


class OpenSandboxBackend(BaseSandbox):
    """封装 ``SandboxSync`` 的 deepagents 沙箱后端。"""

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
        """新建并拥有一个沙箱;省略 ``connection_config`` 时 SDK 读环境变量。"""
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
        """连接已运行的沙箱;不拥有它,:meth:`close` 不会终止它。"""
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
        """执行命令;错误转为非零退出码,绝不抛异常。"""
        timeout = timeout if timeout is not None else self._default_timeout
        opts = (
            RunCommandOpts(timeout=timedelta(seconds=timeout))
            if timeout is not None and timeout > 0
            else None
        )
        try:
            execution = self._sandbox.commands.run(command, opts=opts)
        except Exception as exc:
            logger.debug("execute failed: %r", command, exc_info=exc)
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
            )
        return ExecuteResponse(output=_output(execution), exit_code=_exit_code(execution))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """经 SDK 文件传输原生读取;语义与基类脚本逐字节一致(见 tests/test_read_parity.py)。"""
        try:
            raw = self._sandbox.files.read_bytes(file_path)
        except Exception as exc:
            logger.debug("read failed: %s", file_path, exc_info=exc)
            return _read_error(file_path, exc)
        return _read(file_path, raw, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [self._put(path, data) for path, data in files]

    def _put(self, path: str, data: bytes) -> FileUploadResponse:
        try:
            self._sandbox.files.write_file(path, data)
            return FileUploadResponse(path=path)
        except Exception as exc:
            parent = posixpath.dirname(path)
            if (
                parent not in ("", "/")
                and self.execute(f"mkdir -p {shlex.quote(parent)}").exit_code == 0
            ):
                try:
                    self._sandbox.files.write_file(path, data)
                    return FileUploadResponse(path=path)
                except Exception as retry_exc:
                    exc = retry_exc
            logger.debug("upload failed: %s", path, exc_info=exc)
            return FileUploadResponse(path=path, error=_classify(exc) or str(exc))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [self._get(path) for path in paths]

    def _get(self, path: str) -> FileDownloadResponse:
        try:
            return FileDownloadResponse(path=path, content=self._sandbox.files.read_bytes(path))
        except Exception as exc:
            logger.debug("download failed: %s", path, exc_info=exc)
            return FileDownloadResponse(path=path, error=_classify(exc) or str(exc))

    def close(self) -> None:
        """仅当拥有沙箱时终止并释放;否则为空操作。"""
        if not self._owns_sandbox:
            return
        try:
            self._sandbox.kill()
        except SandboxException as exc:
            logger.warning("kill sandbox %s failed: %s", self._sandbox.id, exc)
        finally:
            self._sandbox.close()

    def __enter__(self) -> OpenSandboxBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"OpenSandboxBackend(id={self._sandbox.id!r}, owns_sandbox={self._owns_sandbox})"


def _output(execution: Execution) -> str:
    """按时间戳合并 stdout/stderr;无输出必须返回空串(解析器契约)。"""
    messages = sorted(
        [*execution.logs.stdout, *execution.logs.stderr], key=lambda m: m.timestamp
    )
    return "\n".join(m.text.rstrip("\n") for m in messages)


def _exit_code(execution: Execution) -> int:
    """服务端缺省退出码时按有无 error 推断,保证 write 预检可靠。"""
    if execution.exit_code is not None:
        return execution.exit_code
    return 1 if execution.error is not None else 0


def _classify(exc: Exception) -> str | None:
    """SDK 异常 → deepagents 文件错误码;无法识别返回 ``None``。"""
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status == 404 or "not found" in message or "no such file" in message:
        return FILE_NOT_FOUND
    if status == 403 or "permission denied" in message:
        return PERMISSION_DENIED
    return None


def _read_error(path: str, exc: Exception) -> ReadResult:
    detail = _classify(exc) or f"{type(exc).__name__}: {exc}"
    return ReadResult(error=f"File '{path}': {detail}")


def _binary(path: str, raw: bytes) -> ReadResult:
    if len(raw) > MAX_BINARY_BYTES:
        return ReadResult(
            error=f"File '{path}': Binary file exceeds maximum preview size of {MAX_BINARY_BYTES} bytes"
        )
    return ReadResult(
        file_data=FileData(content=base64.b64encode(raw).decode("ascii"), encoding="base64")
    )


def _read(path: str, raw: bytes, offset: int, limit: int) -> ReadResult:
    """字节 → ReadResult,复刻基类脚本的空文件/二进制/换行/分页/截断语义。"""
    if not raw:
        return ReadResult(file_data=FileData(content=EMPTY_CONTENT_WARNING, encoding="utf-8"))
    if _get_file_type(path) != "text":
        return _binary(path, raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("non-utf8 text file %s; returning base64", path)
        return _binary(path, raw)

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    offset, limit = int(offset), int(limit)
    if not lines or offset >= len(lines):
        return ReadResult(
            error=f"File '{path}': Line offset {offset} exceeds file length ({len(lines)} lines)"
        )

    content = "\n".join(lines[offset : offset + limit])
    encoded = content.encode("utf-8")
    cap = MAX_OUTPUT_BYTES - len(TRUNCATION_MSG.encode("utf-8"))
    if len(encoded) > cap:
        content = encoded[:cap].decode("utf-8", errors="ignore") + TRUNCATION_MSG
    return ReadResult(file_data=FileData(content=content, encoding="utf-8"))
