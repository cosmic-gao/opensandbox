"""把 deepagents 的文件系统与 Shell 操作运行在阿里 OpenSandbox 沙箱内的执行后端。

本模块提供 :class:`OpenSandboxBackend`——一个具体的 deepagents 后端,让智能体的
文件读写与命令执行发生在 OpenSandbox 沙箱(https://github.com/alibaba/OpenSandbox)内部。

设计思路
--------
deepagents 采用可插拔的后端协议。需要执行 Shell 的后端实现
``SandboxBackendProtocol``。该库提供了抽象基类
:class:`deepagents.backends.sandbox.BaseSandbox`,它已经借助若干“服务端脚本 +
一个 ``execute()`` 原语”实现了**全部**文件操作(``ls``/``read``/``write``/
``edit``/``glob``/``grep``)。因此具体后端只需提供三个原语外加一个 ``id``:

* ``execute(command, *, timeout)`` —— 执行 Shell 命令;
* ``upload_files(files)``          —— 把字节写入路径;
* ``download_files(paths)``        —— 从路径读取字节;
* ``id``                           —— 稳定标识。

:class:`OpenSandboxBackend` 把这三个原语映射到 OpenSandbox 的**同步** SDK
(``opensandbox.SandboxSync``):

======================  ===============================================
deepagents 原语          OpenSandbox SDK 调用
======================  ===============================================
``execute``             ``sandbox.commands.run(cmd, opts=...)``
``upload_files``        ``sandbox.files.write_file(path, data: bytes)``
``download_files``      ``sandbox.files.read_bytes(path) -> bytes``
======================  ===============================================

由于 ``BaseSandbox`` 从 ``execute`` 派生出 ``ls``/``read``/``edit``/``glob``/``grep``
(并从 ``upload_files`` 派生出 ``write``),这些操作无需额外实现,且直接复用
OpenSandbox 服务端成熟的分页、CRLF 处理与二进制检测能力。

同步与异步
----------
本后端是同步的,封装 ``SandboxSync``,但同样适用于异步智能体:
``BaseSandbox``/``SandboxBackendProtocol`` 提供了异步包装
(``aexecute``/``aupload_files``/...),通过 ``asyncio.to_thread`` 把同步调用
卸载到工作线程,因此事件循环不会被阻塞。
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

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注的导入
    from opensandbox.config import ConnectionConfigSync
    from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger(__name__)

__all__ = ["OpenSandboxBackend"]

# ``create`` 辅助方法使用的默认镜像。Python 镜像对编码类智能体是合理的默认值;
# 可通过 ``create(image=...)`` 覆盖。
DEFAULT_IMAGE = "python:3.11"

# ``create`` 创建的沙箱的默认远端存活时长。到期后若未续期则自动终止,
# 从而避免后端泄漏导致容器长期不回收。
DEFAULT_LIFETIME = timedelta(minutes=30)


class OpenSandboxBackend(BaseSandbox):
    """由 OpenSandbox 沙箱支撑的 deepagents 后端。

    该后端既可以封装一个你自行管理的 :class:`~opensandbox.SandboxSync` 实例,
    也可以通过 :meth:`create` / :meth:`connect` 类方法帮你创建或连接沙箱。

    Args:
        sandbox: 一个就绪的 ``SandboxSync`` 实例。
        owns_sandbox: 若为 ``True``,:meth:`close` 会 ``kill()`` 远端沙箱并
            释放本地资源;由 :meth:`create` 自动置为 ``True``。封装你自己拥有
            的沙箱时保持 ``False``,以免关闭后端时误销毁沙箱。
        default_timeout: 调用 ``execute`` 未显式指定 ``timeout`` 时使用的
            默认单命令超时(秒)。``None`` 表示“客户端不施加超时”(服务端仍
            可能有自己的限制)。

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents_opensandbox import OpenSandboxBackend

        with OpenSandboxBackend.create(image="python:3.11") as backend:
            agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
            result = agent.invoke({"messages": [{"role": "user",
                     "content": "创建 fib.py,打印前 10 个斐波那契数,然后运行它。"}]})
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

    # -- 构造辅助方法 --------------------------------------------------------

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
        """新建一个 OpenSandbox 沙箱并封装它。

        返回的后端“拥有”该沙箱::meth:`close`(或退出 ``with`` 块)会将其终止。

        Args:
            image: 容器镜像引用(如 ``"python:3.11"``),或用于私有仓库鉴权的
                ``SandboxImageSpec``。
            connection_config: 连接配置。省略时 SDK 会读取环境变量
                ``OPEN_SANDBOX_DOMAIN``(默认 ``localhost:8080``)与
                ``OPEN_SANDBOX_API_KEY``。
            timeout: 远端沙箱自动终止前的存活时长;传 ``None`` 则要求显式清理。
            default_timeout: ``execute`` 的默认单命令超时(秒)。
            **create_kwargs: 原样透传给 ``SandboxSync.create``(如 ``env=``、
                ``resource=``、``metadata=``、``network_policy=``)。
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
        """按 id 连接到一个已在运行的沙箱并封装它。

        返回的后端**不拥有**该沙箱,因此 :meth:`close` 不会终止远端沙箱
        (其生命周期由别处管理)。
        """
        sandbox = SandboxSync.connect(
            sandbox_id,
            connection_config=connection_config,
            **connect_kwargs,  # type: ignore[arg-type]
        )
        return cls(sandbox, owns_sandbox=False, default_timeout=default_timeout)

    # -- 自省 ----------------------------------------------------------------

    @property
    def sandbox(self) -> SandboxSync:
        """底层的 OpenSandbox ``SandboxSync`` 实例。"""
        return self._sandbox

    @property
    def id(self) -> str:
        """后端的稳定标识(即 OpenSandbox 沙箱 id)。"""
        return self._sandbox.id

    # -- 核心原语:Shell 执行 ------------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """在沙箱内运行 ``command``,返回合并后的输出与退出码。

        这是 ``BaseSandbox`` 构建所有文件操作的唯一原语,因此必须如实反映输出
        与退出码(``write`` 的前置检查依赖非零退出码来判断“文件已存在”)。

        绝不抛异常:任何传输/SDK 错误都会转成带非零退出码的 ``ExecuteResponse``,
        与内置后端的约定保持一致。
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        opts: RunCommandOpts | None = None
        # 超时为 0(或负数)表示“客户端不设超时”,此时不构造 opts,
        # 让服务端应用自身策略,而非施加一个 0 秒的截止时间。
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

    # -- 核心原语:文件上传 --------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """把若干 ``(path, bytes)`` 写入沙箱,支持部分成功。

        每个文件独立写入,单个失败不会中断其余文件。若因父目录缺失而写入失败,
        则先用 ``mkdir -p`` 创建目录再重试一次(部分 execd 部署在上传时不会
        自动创建父目录)。
        """
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

    # -- 核心原语:文件下载 --------------------------------------------------

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """逐个从沙箱读取原始字节,支持部分成功。"""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:  # noqa: BLE001 - 逐个文件上报,绝不抛出
                logger.debug("download failed for %s", path, exc_info=exc)
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_classify_error(exc) or str(exc),
                    )
                )
        return responses

    # -- 生命周期 ------------------------------------------------------------

    def close(self) -> None:
        """释放资源。

        对于经 :meth:`create` 创建(``owns_sandbox=True``)的沙箱,会终止远端
        沙箱并关闭本地 HTTP 传输;对于封装的、你自己拥有的沙箱,则为空操作,
        以保持沙箱继续运行。
        """
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
        return (
            f"OpenSandboxBackend(id={self._sandbox.id!r}, "
            f"owns_sandbox={self._owns_sandbox})"
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _combine_output(execution: Execution) -> str:
    """把 stdout 与 stderr 按时间戳合并为单一输出流。

    无输出时返回空字符串。这一点很关键:``BaseSandbox`` 的解析器把空输出视为
    “无结果”(例如 grep 无匹配),因此不能用 ``"<no output>"`` 之类的哨兵值替代。
    每条消息去掉尾部换行后再以 ``"\\n"`` 拼接,避免流式分片产生多余空行。
    """
    messages = [*execution.logs.stdout, *execution.logs.stderr]
    # 稳定排序:时间戳相同时保持 stdout 在 stderr 之前。
    messages.sort(key=lambda m: m.timestamp)
    return "\n".join(msg.text.rstrip("\n") for msg in messages)


def _exit_code(execution: Execution) -> int:
    """返回确定的退出码;服务端缺省时进行推断。

    某些运行时对流式前台命令不回传 ``exit_code``。若把未知码默认为 ``0``,会让
    ``write`` 的前置检查(“文件已存在”会以退出码 1 结束)失效,因此在存在执行
    错误时推断为失败。
    """
    if execution.exit_code is not None:
        return execution.exit_code
    return 1 if execution.error is not None else 0


def _classify_error(exc: Exception) -> str | None:
    """把 SDK 异常映射为 deepagents 标准化文件错误码。

    当失败可识别为“未找到 / 权限拒绝”时,返回对应的 ``FileOperationError``
    字面量,否则返回 ``None``,让调用方回退到原始错误字符串。
    """
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status == 404 or "not found" in message or "no such file" in message:
        return FILE_NOT_FOUND
    if status == 403 or "permission denied" in message:
        return PERMISSION_DENIED
    return None
