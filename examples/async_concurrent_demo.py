"""并发驱动多个 OpenSandbox 沙箱(原生异步),演示高并发吞吐。无需 LLM。

这是评测 / RL 场景的核心形态:多个沙箱同时在途,靠事件循环而非线程池承载并发。
每个任务独占一个沙箱,全部用 ``asyncio.gather`` 并发执行。

前置条件:一个在运行的 OpenSandbox 服务端(见 raw_backend_demo.py);
环境变量 OPEN_SANDBOX_DOMAIN / OPEN_SANDBOX_API_KEY。

运行:
    python examples/async_concurrent_demo.py
"""

from __future__ import annotations

import asyncio
import time

from deepagents_opensandbox import AsyncOpenSandboxBackend

N = 8  # 并发沙箱数量


async def one_task(index: int) -> tuple[int, str, int | None]:
    # 每个任务创建并拥有自己的沙箱,退出时自动 kill。
    async with await AsyncOpenSandboxBackend.create(image="python:3.11") as backend:
        res = await backend.aexecute(f"python -c 'print(sum(range({index} * 100000)))'")
        return index, res.output.strip(), res.exit_code


async def main() -> None:
    start = time.perf_counter()
    # N 个沙箱的创建与执行全部并发在途——事件循环调度,不占线程池。
    results = await asyncio.gather(*(one_task(i) for i in range(1, N + 1)))
    elapsed = time.perf_counter() - start

    for index, output, exit_code in results:
        print(f"task {index}: exit={exit_code} out={output}")
    print(f"\n{N} 个沙箱并发完成,用时 {elapsed:.1f}s(顺序执行会慢得多)。")


if __name__ == "__main__":
    asyncio.run(main())
