# deepagents-opensandbox

一个自定义的 [**deepagents**](https://github.com/langchain-ai/deepagents) 后端,让深度智能体的**文件系统与 Shell 操作运行在阿里 [OpenSandbox](https://github.com/alibaba/OpenSandbox) 沙箱内**。

只需把一个对象传给 `create_deep_agent`,即可让智能体在隔离的 Linux 容器里读写文件、运行代码,而不触碰宿主机。

```python
from deepagents import create_deep_agent
from deepagents_opensandbox import OpenSandboxBackend

with OpenSandboxBackend.create(image="python:3.11") as backend:
    agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
    agent.invoke({"messages": [{"role": "user",
        "content": "Write primes.py that prints primes < 50, run it, show output."}]})
    # ls / read_file / write_file / edit_file / glob / grep / execute 全部在沙箱内执行
```

---

## 背景:两个项目

**OpenSandbox**(阿里巴巴,Apache-2.0)是面向 AI 应用的通用沙箱平台。它定义了沙箱协议(生命周期 + 执行 API),可运行在 Docker 或 Kubernetes 之上(支持 gVisor/Kata/Firecracker),并提供多语言 SDK(Python、JS/TS、Java/Kotlin、C#、Go)。其核心执行模块是**命令执行、文件系统、代码解释器**——正是智能体安全运行不受信任代码所需的能力。

**deepagents**(LangChain)基于可插拔的**后端(backend)**抽象来构建“深度智能体”。所有文件系统工具(`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`)都经由 `BackendProtocol`;能额外执行 Shell 的后端实现 `SandboxBackendProtocol`(增加 `execute`)。内置后端包括内存态、本地磁盘、LangGraph store、以及远端沙箱。

> 说明:带“backend”概念的是 deepagents(LangChain),不是阿里。阿里同名的 *通义 DeepResearch* 是一个 Web-Agent 模型,并非后端框架。本包做的是 **deepagents ⇄ OpenSandbox** 的桥接。

本包即这座桥:一个存储与执行都发生在 OpenSandbox 沙箱内的 `SandboxBackendProtocol` 实现。

## 工作原理

deepagents 提供了抽象基类 `deepagents.backends.sandbox.BaseSandbox`,它已经借助若干“服务端脚本 + 一个 `execute()` 原语”实现了**全部**文件操作。具体后端只需提供三个原语外加一个 id:

| deepagents 原语 | OpenSandbox SDK 调用 |
| --- | --- |
| `execute(command, *, timeout)` | `sandbox.commands.run(cmd, opts=...)` → 合并 stdout/stderr + 退出码 |
| `upload_files([(path, bytes)])` | `sandbox.files.write_file(path, data)` |
| `download_files([path])` | `sandbox.files.read_bytes(path)` → `bytes` |
| `id` | `sandbox.id` |

`OpenSandboxBackend` 就是把这三个原语实现到 OpenSandbox 的**同步** SDK 上。`ls` / `edit` / `glob` / `grep`(由 `execute` 派生)与 `write`(由 `upload_files` 派生)因而无需额外实现。

在此之上,`read` 做了**原生覆盖**(与官方 `LangSmithSandbox` 同一做法):不再借 `execute()` 在沙箱内跑 python3 脚本、内容经 stdout 以 JSON 回传,而是经 SDK 文件传输直取字节,分页 / CRLF 归一 / 二进制检测 / 截断语义在客户端逐字节复刻(有针对基类脚本的差分测试保证)。每次读取因此少 spawn 一个进程、读取路径不依赖镜像内有 python3、也绕开了 stdout 这条脆弱通道。取舍:整文件下载后再分页,超大文件比服务端分页多传输。

```
create_deep_agent(backend=OpenSandboxBackend)
        │  ls / read / write / edit / glob / grep / execute  (deepagents 工具)
        ▼
BaseSandbox   ── 派生文件操作自 ──▶  execute() / upload_files() / download_files()
        │                            (read 原生覆盖,经文件传输直取,不走派生)
        ▼
OpenSandboxBackend  ── 映射到 ──▶  SandboxSync.commands.run / files.write_file / files.read_bytes
        ▼
OpenSandbox 服务端(Docker / Kubernetes 运行时)
```

## 选择后端:同步 vs 异步

本包提供两个后端,机制相同(都继承 `BaseSandbox`),差别在于底层用同步还是异步 SDK:

| 后端 | 封装 | 适用 | 说明 |
| --- | --- | --- | --- |
| `OpenSandboxBackend` | `SandboxSync`(同步) | 通用默认;单智能体、交互式 | sync/async 智能体都能用;配异步智能体时,deepagents 通过 `asyncio.to_thread` 把阻塞调用卸载到线程池 |
| `AsyncOpenSandboxBackend` | `Sandbox`(异步) | **高并发 / 批量评测 / RL** | 原生协程,仅重写 3 个异步原语即让整个异步接口走原生异步;不占线程池,并发可达数千,取消/超时更干净。**仅限异步智能体(`.ainvoke()`)** |

一句话:日常单智能体用 `OpenSandboxBackend`;在一个进程里同时跑大量沙箱/子智能体时用 `AsyncOpenSandboxBackend`。

> 原理:deepagents 的异步文件工具会调用 `als`/`aread`/`aexecute` 等;`BaseSandbox` 又把这些异步文件操作都转发到 `self.aexecute`/`self.aupload_files`。因此 `AsyncOpenSandboxBackend` 只需原生实现 `aexecute`/`aupload_files`/`adownload_files`(批量条目 `asyncio.gather` 并发、结果保序),整个异步接口便都是原生异步;`aread` 同样做了原生覆盖(直走异步文件传输)。它是 async-only:同步原语会抛 `NotImplementedError`。

## 安装

```bash
pip install deepagents opensandbox      # 运行时依赖
pip install -e .                        # 本包(源码安装)
# 或者,连同 LLM 示例与测试一起:
pip install -e ".[examples,dev]"
```

要求 Python ≥ 3.11(开发环境使用最新的 3.14)。

## 前置条件:一个 OpenSandbox 服务端

SDK 需要与 OpenSandbox 服务端通信。本地 Docker 运行时:

```bash
uvx opensandbox-server init-config ~/.sandbox.toml --example docker
uvx opensandbox-server            # 默认监听 localhost:8080
```

通过环境变量配置连接(未显式传入 `connection_config` 时自动读取):

```bash
export OPEN_SANDBOX_DOMAIN=localhost:8080   # 默认值
export OPEN_SANDBOX_API_KEY=...             # 若服务端要求鉴权
```

参见 [`.env.example`](.env.example)。

## 用法

### 1. 让后端拥有一个新建沙箱(推荐)

```python
from deepagents_opensandbox import OpenSandboxBackend

with OpenSandboxBackend.create(image="python:3.11", default_timeout=120) as backend:
    ...  # 退出 with 块时沙箱自动终止
```

`create(...)` 会把额外参数透传给 `SandboxSync.create`(`env=`、`resource=`、`metadata=`、`network_policy=` 等)。

### 2. 封装你自行管理的沙箱

```python
from opensandbox import SandboxSync
from deepagents_opensandbox import OpenSandboxBackend

sandbox = SandboxSync.create("python:3.11")
backend = OpenSandboxBackend(sandbox)        # owns_sandbox=False → close() 不会 kill 它
# ... 与智能体一起使用 ...
sandbox.kill(); sandbox.close()              # 生命周期由你掌控
```

### 3. 按 id 连接到已运行的沙箱

```python
backend = OpenSandboxBackend.connect("sbx-abc123")
```

### 与深度智能体配合

```python
from deepagents import create_deep_agent

agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
agent.invoke({"messages": [{"role": "user", "content": "..."}]})   # 同步
await agent.ainvoke({"messages": [...]})                            # 异步同样可用
```

**异步说明:** `OpenSandboxBackend`(同步)也适用于异步智能体——`BaseSandbox` 提供异步包装(`aexecute`、`aupload_files`……),通过 `asyncio.to_thread` 把阻塞的 SDK 调用卸载到工作线程,因此事件循环不会被阻塞。若追求高并发,见下面的原生异步后端。

### 4. 高并发:原生异步后端

```python
import asyncio
from deepagents import create_deep_agent
from deepagents_opensandbox import AsyncOpenSandboxBackend

async def run_task(task: str) -> str:
    async with await AsyncOpenSandboxBackend.create(image="python:3.11") as backend:
        agent = create_deep_agent(model="openai:gpt-5.5", backend=backend)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content

# 多个智能体各占一个沙箱,并发 fan-out(评测 / RL 的典型形态)
answers = asyncio.run(asyncio.gather(*(run_task(t) for t in tasks)))
```

`AsyncOpenSandboxBackend` 的 `create`/`connect` 是协程(需 `await`),生命周期用 `aclose()` / `async with`。它只服务 `.ainvoke()`;误用同步接口会抛 `NotImplementedError`。

## `OpenSandboxBackend` API

| 成员 | 说明 |
| --- | --- |
| `OpenSandboxBackend(sandbox, *, owns_sandbox=False, default_timeout=None)` | 封装已有的 `SandboxSync`。 |
| `OpenSandboxBackend.create(image="python:3.11", *, connection_config=None, timeout=30min, default_timeout=None, **create_kwargs)` | 新建并拥有一个沙箱。 |
| `OpenSandboxBackend.connect(sandbox_id, *, connection_config=None, ...)` | 连接到运行中的沙箱(不拥有)。 |
| `.sandbox` / `.id` | 底层 `SandboxSync` / 其 id。 |
| `.execute(command, *, timeout=None)` | 执行 Shell 命令 → `ExecuteResponse(output, exit_code, truncated)`。 |
| `.read(path, offset=0, limit=2000)` | **原生覆盖**:经 SDK 文件传输直取字节,客户端复刻基类分页语义。 |
| `.upload_files([(path, bytes)])` / `.download_files([path])` | 批量字节传输(支持部分成功)。 |
| `.close()` / 上下文管理器 | **仅当**拥有时才 kill + 释放;否则为空操作。 |
| 继承自 `BaseSandbox` | `ls`、`write`、`edit`、`glob`、`grep`(及异步 `a*`)。 |

`AsyncOpenSandboxBackend` 接口与之对应,但为异步:`await create(...)` / `await connect(...)`、`aexecute` / `aread` / `aupload_files` / `adownload_files`(批量并发、保序)、`aclose()` / `async with`;派生的 `als`/`awrite`/`aedit`/`aglob`/`agrep` 均为原生异步。同步原语(`execute` 等)会抛 `NotImplementedError`。

### 行为说明

- **绝不抛异常:** 原语把 SDK/传输错误转成结果对象(非零退出码 / 逐文件 `error`),符合 deepagents 约定。
- **空输出 → `""`**(而非哨兵值),以保证 `grep` 无匹配时能被正确解析。
- **退出码:** 取自服务端;若运行时未回传,则在执行报错时推断为 `1`,否则 `0`(保证 `write` 的“文件已存在则失败”前置检查可靠)。
- **父目录缺失:** 若因父目录不存在导致写入失败,`upload_files` 会先 `mkdir -p` 再重试一次。
- **错误映射:** 未找到 → `file_not_found`,权限拒绝 → `permission_denied`(deepagents 的 `FileOperationError` 字面量)。
- **read 原生覆盖:** 与基类脚本语义逐字节一致——空文件返回“空内容提醒”;扩展名归类为图像/音视频、或 UTF-8 解码失败的文件走 base64(上限 500 KiB,超限报错);文本做通用换行归一(CRLF 与孤立 `\r` 折叠为 `\n`)后按 `offset`/`limit` 分页,页超 500 KiB 截断并追加续读提示。相对路径由 SDK 按工作目录解析(基类契约的超集)。
- **write 无需覆盖:** 基类 `write` = 存在性预检(经 `execute`)+ `upload_files` 传输;本后端的 `upload_files` 已是原生文件传输,内容不过 shell,故形态与官方 `LangSmithSandbox` 的覆盖等价。

## 测试

单元测试使用内存版假沙箱;`tests/test_read_parity.py` 另以差分方式验证原生 `read` 与基类服务端脚本逐字节等价(进程内执行同一段脚本源码比对)。两者都无需服务端,跨平台:

```bash
pip install -e ".[dev]"
pytest
```

针对真实服务端的完整端到端链路:

```bash
export RUN_OPENSANDBOX_INTEGRATION=1
export OPEN_SANDBOX_DOMAIN=localhost:8080
pytest -k integration
```

## 示例

- [`examples/raw_backend_demo.py`](examples/raw_backend_demo.py) —— 不用 LLM key,直接驱动后端,演示 `execute`/`write`/`read`/`edit`/`ls`/`grep`/`glob`。
- [`examples/quickstart_agent.py`](examples/quickstart_agent.py) —— 一个完整的深度智能体,其工具全部在 OpenSandbox 内运行。
- [`examples/async_concurrent_demo.py`](examples/async_concurrent_demo.py) —— 不用 LLM key,并发驱动多个异步沙箱,演示高并发吞吐。
- [`examples/async_agent.py`](examples/async_agent.py) —— 多个深度智能体各占一个沙箱,`asyncio.gather` 并发 fan-out(评测 / RL 形态)。

## 许可证

Apache-2.0(与 OpenSandbox 保持一致)。
