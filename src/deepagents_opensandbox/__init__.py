"""deepagents-opensandbox:让 deepagents 运行在阿里 OpenSandbox 沙箱内。

对外暴露两个 deepagents 沙箱后端,其文件系统与 Shell 操作都在 OpenSandbox 沙箱内执行:

* :class:`OpenSandboxBackend` —— 同步(封装 ``SandboxSync``),通用默认,sync/async 皆可跑;
* :class:`AsyncOpenSandboxBackend` —— 原生异步(封装 ``Sandbox``),用于高并发/评测/RL。

快速上手:
    >>> from deepagents import create_deep_agent
    >>> from deepagents_opensandbox import OpenSandboxBackend
    >>> with OpenSandboxBackend.create(image="python:3.11") as backend:
    ...     agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
    ...     agent.invoke({"messages": [{"role": "user", "content": "..."}]})
"""

from deepagents_opensandbox.async_backend import AsyncOpenSandboxBackend
from deepagents_opensandbox.backend import OpenSandboxBackend

__all__ = ["AsyncOpenSandboxBackend", "OpenSandboxBackend", "__version__"]
__version__ = "0.1.0"
