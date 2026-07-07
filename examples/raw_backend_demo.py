"""不借助 LLM,直接驱动 OpenSandboxBackend,验证 deepagents <-> OpenSandbox 全链路。

下面每个方法都是 deepagents 的 ``BackendProtocol`` 操作,由 ``BaseSandbox`` 基于三个
原语(execute/upload_files/download_files)派生。

前置条件:一个在运行的 OpenSandbox 服务端(本地:``uvx opensandbox-server``);
环境变量 OPEN_SANDBOX_DOMAIN(默认 localhost:8080)、OPEN_SANDBOX_API_KEY。

运行:
    python examples/raw_backend_demo.py
"""

from __future__ import annotations

from deepagents_opensandbox import OpenSandboxBackend

WORKDIR = "/workspace/demo"

# 写入沙箱并运行的示例程序(被执行的程序内容,保持英文)。
SCRIPT = """\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


if __name__ == "__main__":
    # print the first 10 Fibonacci numbers
    print(list(fib(10)))
"""


def main() -> None:
    # create() 新建并拥有沙箱,退出 with 时自动 kill。
    with OpenSandboxBackend.create(image="python:3.11") as backend:
        print(f"sandbox id: {backend.id}\n")

        res = backend.execute(f"mkdir -p {WORKDIR} && echo ready")
        print("execute:", res.output, f"(exit={res.exit_code})")

        w = backend.write(f"{WORKDIR}/fib.py", SCRIPT)
        print("write:", w.path or w.error)

        run = backend.execute(f"cd {WORKDIR} && python fib.py")
        print("run fib.py:", run.output, f"(exit={run.exit_code})")

        r = backend.read(f"{WORKDIR}/fib.py")
        print("read:", r.error or r.file_data["content"].splitlines()[0])

        e = backend.edit(f"{WORKDIR}/fib.py", "first 10", "first 15")
        print("edit:", e.error or f"{e.occurrences} occurrence(s)")

        ls = backend.ls(WORKDIR)
        print("ls:", [entry["path"] for entry in (ls.entries or [])])

        g = backend.grep("range", WORKDIR)
        print("grep:", [(m["path"], m["line"]) for m in (g.matches or [])])

        gl = backend.glob("**/*.py", WORKDIR)
        print("glob:", [m["path"] for m in (gl.matches or [])])


if __name__ == "__main__":
    main()
