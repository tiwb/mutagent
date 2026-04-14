"""执行引擎 — 白名单 builtins + REPL 语义。"""

import ast
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


# 安全内置函数白名单
_SAFE_BUILTINS = {
    'len': len, 'range': range, 'enumerate': enumerate,
    'zip': zip, 'map': map, 'filter': filter,
    'sorted': sorted, 'reversed': reversed,
    'min': min, 'max': max, 'sum': sum,
    'any': any, 'all': all, 'abs': abs, 'round': round,
    'isinstance': isinstance, 'print': print, 'repr': repr,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'bytes': bytes,
    'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
    'hash': hash, 'id': id,
    'iter': iter, 'next': next,
    'callable': callable,
    'Exception': Exception, 'ValueError': ValueError,
    'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
    'ZeroDivisionError': ZeroDivisionError,
    'NotImplementedError': NotImplementedError,
    'OverflowError': OverflowError,
    'True': True, 'False': False, 'None': None,
}

# type() 限制为单参数（类型查询），禁止三参数动态建类
_builtin_type = type


def _safe_type(*args):
    if len(args) != 1:
        raise TypeError("type() takes 1 argument")
    return _builtin_type(args[0])


_SAFE_BUILTINS['type'] = _safe_type


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
    """
    # 构建 globals：安全 builtins + 注入的命名空间
    globals_dict = {}
    globals_dict['__builtins__'] = dict(_SAFE_BUILTINS)
    globals_dict.update(namespace)

    # locals 用于 REPL 状态保持
    locals_dict = state if state is not None else {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            # REPL 语义：最后一条表达式自动返回值
            tree = ast.parse(code, '<pysandbox>', 'exec')

            # AST 检查：禁止 import 和 class
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise SyntaxError("import statements are not supported")
                if isinstance(node, ast.ClassDef):
                    raise SyntaxError("class definitions are not supported")

            last_expr_value = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                if tree.body:
                    exec(compile(ast.Module(body=tree.body, type_ignores=[]),
                                 '<pysandbox>', 'exec'),
                         globals_dict, locals_dict)
                last_expr_value = eval(
                    compile(ast.Expression(body=last_expr.value),
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
            "stderr": stderr_buf.getvalue(),
        }

    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "traceback": ""}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}
