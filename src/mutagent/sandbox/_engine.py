"""执行引擎 — 黑名单 builtins + REPL 语义。"""

import ast
import builtins
import io
import traceback as _tb_mod
from typing import Any


# 必须排除的 builtins（安全边界）
_BLOCKED_BUILTINS = frozenset({
    'eval', 'exec', 'compile', '__import__',
    'open', 'breakpoint', 'input',
    'exit', 'quit', 'help',
})

# 从 builtins 模块取全集，排除黑名单
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if not name.startswith('_') and name not in _BLOCKED_BUILTINS
}
# 保留 __build_class__（class 语句需要）和 __name__
_SAFE_BUILTINS['__build_class__'] = builtins.__build_class__
_SAFE_BUILTINS['__name__'] = '<pysandbox>'


def execute(code: str, namespace: dict[str, Any],
            state: dict[str, Any] | None = None) -> dict[str, Any]:
    """在受限环境中执行 Python 代码。

    Args:
        code: 要执行的 Python 代码
        namespace: 注入的函数和对象（命名空间对象、自省函数等）
        state: 跨步骤共享状态（REPL 模式）。None 时变量存入临时 dict

    Returns:
        {"result": Any, "stdout": str, "stderr": str}
        失败时返回 {"error": str, "traceback": str}

    Note:
        ``stderr`` 字段保留键以维持 caller 契约，但当前 sandbox 封闭
        配置下用户代码无法写入 stderr（无 ``sys`` / ``import`` /
        file-like 暴露），该字段始终为空字符串。若未来放开 import
        或在 namespace 注入 stderr-writable 对象（如默认 logger），需
        重新引入 stderr 捕获，参见
        ``docs/specifications/bugfix-sandbox-thread-safety.md`` 的
        「已知前提」一节。
    """
    # 构建 globals：安全 builtins + 注入的命名空间
    globals_dict = {}
    safe_builtins = dict(_SAFE_BUILTINS)
    globals_dict['__builtins__'] = safe_builtins
    globals_dict.update(namespace)

    # locals 用于 REPL 状态保持
    locals_dict = state if state is not None else {}

    stdout_buf = io.StringIO()

    # 注入线程安全的 print，写入私有 buffer。
    # 用闭包捕获本次调用的 stdout_buf，避免 redirect_stdout 替换全局
    # sys.stdout 导致并发线程互相污染。
    def _safe_print(*args, **kwargs):
        kwargs.setdefault('file', stdout_buf)
        builtins.print(*args, **kwargs)

    safe_builtins['print'] = _safe_print

    try:
        # REPL 语义：最后一条表达式自动返回值
        tree = ast.parse(code, '<pysandbox>', 'exec')

        # AST 检查：禁止 import
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SyntaxError("import statements are not supported")

        last_expr_value = None
        last_node = tree.body[-1] if tree.body else None
        if isinstance(last_node, ast.Expr):
            tree.body.pop()
            if tree.body:
                exec(compile(ast.Module(body=tree.body, type_ignores=[]),
                             '<pysandbox>', 'exec'),
                     globals_dict, locals_dict)
            last_expr_value = eval(
                compile(ast.Expression(body=last_node.value),
                        '<pysandbox>', 'eval'),
                globals_dict, locals_dict)
        else:
            exec(compile(tree, '<pysandbox>', 'exec'),
                 globals_dict, locals_dict)

        # 优先使用显式 result 变量
        result = locals_dict.get('result')
        if result is None:
            result = last_expr_value

        return {
            "result": result,
            "stdout": stdout_buf.getvalue(),
            "stderr": "",
        }

    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "traceback": ""}
    except Exception as e:
        tb = _filter_traceback(e)
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}


def _filter_traceback(exc: BaseException) -> str:
    """过滤 traceback，去掉引擎入口帧，保留 <pysandbox> 及其下游所有帧。"""
    frames = _tb_mod.extract_tb(exc.__traceback__)
    # 找到第一个 <pysandbox> 帧，从那里开始保留
    start = next((i for i, f in enumerate(frames) if f.filename == '<pysandbox>'), None)
    if start is None:
        return ""
    kept = frames[start:]
    # 外部函数帧：隐藏文件路径，只保留函数名
    cleaned = []
    for f in kept:
        if f.filename != '<pysandbox>':
            cleaned.append(_tb_mod.FrameSummary('<external>', 0, f.name))
        else:
            cleaned.append(f)
    lines = ["Traceback (most recent call last):"]
    lines.extend(_tb_mod.format_list(cleaned))
    return "".join(lines).rstrip()
