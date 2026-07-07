"""并发运行多个深度智能体,各自在独立的 OpenSandbox 沙箱内工作(原生异步)。

这是批量评测 / RL rollout 的典型形态:N 个智能体各占一个沙箱,用 ``asyncio.gather``
并发跑;由 :class:`AsyncOpenSandboxBackend` 原生异步承载,吞吐远高于同步后端 + 线程池。

前置条件:一个在运行的 OpenSandbox 服务端;环境变量 OPEN_SANDBOX_DOMAIN /
OPEN_SANDBOX_API_KEY,以及一个 LLM key(此处用 OPENAI_API_KEY)。
安装:`pip install "deepagents-opensandbox[examples]"`。

运行:
    python examples/async_agent.py
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from deepagents_opensandbox import AsyncOpenSandboxBackend

# 一批待评测的任务(作为 LLM 输入内容,保持英文)。
TASKS = [
    "Write and run a Python one-liner that prints the factorial of 6.",
    "Write and run a Python script that prints the 10th prime number.",
    "Write and run a Python script that reverses the string 'opensandbox'.",
    "Write and run a Python script that prints the sum of squares from 1 to 100.",
]


async def run_task(task: str, model: str) -> str:
    # 每个智能体独占一个原生异步沙箱,退出时自动 kill。
    async with await AsyncOpenSandboxBackend.create(
        image="python:3.11", default_timeout=120
    ) as backend:
        agent = create_deep_agent(model=model, backend=backend)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content


async def main() -> None:
    model = "openai:gpt-5.5"
    # 全部智能体并发执行——真正的异步 fan-out。
    answers = await asyncio.gather(*(run_task(task, model) for task in TASKS))
    for task, answer in zip(TASKS, answers):
        print(f"- {task}\n  -> {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
