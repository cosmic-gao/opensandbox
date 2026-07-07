"""deepagents-opensandbox:让 deepagents 运行在阿里 OpenSandbox 沙箱内。

对外暴露 :class:`OpenSandboxBackend`——一个 deepagents 沙箱后端,其文件系统
与 Shell 操作在 OpenSandbox 沙箱内执行。

快速上手:
    >>> from deepagents import create_deep_agent
    >>> from deepagents_opensandbox import OpenSandboxBackend
    >>> with OpenSandboxBackend.create(image="python:3.11") as backend:
    ...     agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
    ...     agent.invoke({"messages": [{"role": "user", "content": "..."}]})
"""

from deepagents_opensandbox.backend import OpenSandboxBackend

__all__ = ["OpenSandboxBackend", "__version__"]
__version__ = "0.1.0"
