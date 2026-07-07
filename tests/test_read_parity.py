"""差分测试:原生 read 与 deepagents 基类服务端脚本(_READ_COMMAND_TEMPLATE)逐字节等价。

剥掉 ``python3 -c "…" 2>&1`` 外壳,在进程内 exec 同一段脚本源码并截获 stdout,
经真实解析器 ``_parse_read_output`` 得到基准 ReadResult;与 ``_read`` 对同一文件比对。
无需服务端,跨平台。
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from deepagents.backends.sandbox import _build_read_cmd, _parse_read_output
from deepagents_opensandbox.backend import _read

PREFIX = 'python3 -c "'
SUFFIX = '" 2>&1'

CASES = [
    ("trailing-nl", "a.txt", b"alpha\nbeta\ngamma\n", 0, 2000),
    ("no-trailing-nl", "b.txt", b"alpha\nbeta\ngamma", 0, 2000),
    ("crlf", "dos.txt", b"a\r\nb\r\nc\r\n", 0, 2000),
    ("bare-cr", "mac.txt", b"a\rb\rc", 0, 2000),
    ("empty", "empty.txt", b"", 0, 2000),
    ("single-blank-line", "blank.txt", b"\n", 0, 2000),
    ("offset-limit", "page.txt", b"l1\nl2\nl3\nl4\nl5\n", 1, 2),
    ("offset-eof", "eof.txt", b"only\n", 5, 2000),
    ("offset-exact-eof", "exact.txt", b"x\ny\n", 2, 2000),
    ("cjk-emoji", "u.txt", "中文\n🚀 emoji\n".encode(), 0, 2000),
    ("nul-text", "nul.bin", b"\x00\x01ok\n", 0, 2000),
    ("png-small", "img.png", b"\x89PNG\r\n\x1a\n\x00\x01", 0, 2000),
    ("png-over-cap", "big.png", b"\x00" * (500 * 1024 + 1), 0, 2000),
    ("truncate-600kb", "huge.txt", ("\n".join(["x" * 1000] * 600)).encode(), 0, 2000),
    (
        "truncate-offset",
        "huge2.txt",
        ("\n".join(f"L{i:04d}" + "y" * 995 for i in range(1200))).encode(),
        600,
        2000,
    ),
]


def template(path: str, offset: int, limit: int):
    cmd = _build_read_cmd(path, offset, limit)
    inner = cmd[len(PREFIX) : cmd.rfind(SUFFIX)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        exec(compile(inner, "<read-template>", "exec"), {"__name__": "__main__"})
    return _parse_read_output(buf.getvalue(), path)


@pytest.mark.parametrize(("name", "fname", "data", "offset", "limit"), CASES, ids=[c[0] for c in CASES])
def test_read_matches_template(tmp_path: Path, name, fname, data, offset, limit):
    file = tmp_path / fname
    file.write_bytes(data)
    path = str(file).replace("\\", "/")

    base = template(path, offset, limit)
    native = _read(path, data, offset, limit)

    assert native.error == base.error
    assert (native.file_data is None) == (base.file_data is None)
    if base.file_data is not None:
        assert native.file_data["content"] == base.file_data["content"]
        assert native.file_data.get("encoding") == base.file_data.get("encoding")
