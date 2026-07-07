"""不借助 LLM,直接驱动 OpenSandboxBackend。

用于验证 deepagents <-> OpenSandbox 的完整链路:下面每个方法都是 deepagents 的
``BackendProtocol`` 操作,由 ``BaseSandbox`` 基于 :class:`OpenSandboxBackend` 实现的
三个原语(``execute``/``upload_files``/``download_files``)派生而来。

前置条件:
  * 一个在运行的 OpenSandbox 服务端。本地 Docker 运行时:
      uvx opensandbox-server init-config ~/.sandbox.toml --example docker
      uvx opensandbox-server
  * 环境变量:OPEN_SANDBOX_DOMAIN(默认 localhost:8080)、OPEN_SANDBOX_API_KEY。

运行:
    python examples/raw_backend_demo.py
"""

from __future__ import annotations

from deepagents_opensandbox import OpenSandboxBackend

WORKDIR = "/workspace/demo"


def main() -> None:
    # create() 会新建一个沙箱并拥有其生命周期(退出时自动 kill)。
    with OpenSandboxBackend.create(image="python:3.11") as backend:
        print(f"sandbox id: {backend.id}\n")

        # 1) execute:在沙箱内执行一条 Shell 命令。
        res = backend.execute(f"mkdir -p {WORKDIR} && echo ready")
        print("execute:", res.output, f"(exit={res.exit_code})")

        # 2) write:创建文件(底层走 upload_files)。
        w = backend.write(f"{WORKDIR}/fib.py", _FIB_SRC)
        print("write:", w.path or w.error)

        # 3) execute:运行刚写入的程序。
        run = backend.execute(f"cd {WORKDIR} && python fib.py")
        print("run fib.py:", run.output, f"(exit={run.exit_code})")

        # 4) read:带行号读回文件。
        r = backend.read(f"{WORKDIR}/fib.py")
        if r.error:
            print("read error:", r.error)
        else:
            print("read (first line):", r.file_data["content"].splitlines()[0])

        # 5) edit:精确字符串替换。
        e = backend.edit(f"{WORKDIR}/fib.py", "first 10", "first 15")
        print("edit occurrences:", e.occurrences if not e.error else e.error)

        # 6) ls:列目录项。
        ls = backend.ls(WORKDIR)
        print("ls:", [entry["path"] for entry in (ls.entries or [])])

        # 7) grep:字面量内容搜索。
        g = backend.grep("range", WORKDIR)
        print("grep matches:", [(m["path"], m["line"]) for m in (g.matches or [])])

        # 8) glob:文件名模式匹配。
        gl = backend.glob("**/*.py", WORKDIR)
        print("glob:", [m["path"] for m in (gl.matches or [])])


# 写入沙箱并运行的示例程序源码(属于被执行的程序内容,保持英文)。
_FIB_SRC = """\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


if __name__ == "__main__":
    # print the first 10 Fibonacci numbers
    print(list(fib(10)))
"""


if __name__ == "__main__":
    main()
