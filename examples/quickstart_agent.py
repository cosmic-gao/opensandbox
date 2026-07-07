"""运行一个 deepagents 深度智能体,其工具全部在 OpenSandbox 内执行。

当把 ``OpenSandboxBackend`` 传给 ``create_deep_agent`` 时,智能体的内置工具
——``ls``、``read_file``、``write_file``、``edit_file``、``glob``、``grep``、
``execute``——都会作用于 OpenSandbox 沙箱而非宿主机。LLM 在隔离容器里写代码、跑代码。

前置条件:
  * 一个在运行的 OpenSandbox 服务端(见 raw_backend_demo.py)。
  * 环境变量:OPEN_SANDBOX_DOMAIN、OPEN_SANDBOX_API_KEY,以及一个 LLM key
    (此处用 OPENAI_API_KEY)。安装:`pip install "deepagents-opensandbox[examples]"`。

运行:
    python examples/quickstart_agent.py
"""

from __future__ import annotations

from deepagents import create_deep_agent

from deepagents_opensandbox import OpenSandboxBackend

# 交给智能体的任务(作为 LLM 输入内容,保持英文)。
TASK = (
    "Write a Python script /workspace/primes.py that prints all prime numbers "
    "below 50, then execute it and report the output."
)


def main() -> None:
    # 后端拥有该沙箱;with 块退出时会 kill 它。
    with OpenSandboxBackend.create(image="python:3.11", default_timeout=120) as backend:
        agent = create_deep_agent(
            # 任意 LangChain 聊天模型 id 均可,如 "anthropic:claude-sonnet-5"。
            model="openai:gpt-5.5",
            backend=backend,
            system_prompt=(
                "You are a coding assistant. You have a sandboxed Linux "
                "filesystem and shell. Write code to files, then run it with "
                "the execute tool and verify the output before answering."
            ),
        )

        result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
