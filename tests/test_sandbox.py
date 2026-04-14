"""mutagent.sandbox 单元测试 — 执行引擎 + 命名空间 + 安全边界。"""

import pytest
from mutagent.sandbox._engine import execute
from mutagent.sandbox._namespace import Namespace, NamespaceRegistry


# ============================================================
# 执行引擎测试
# ============================================================

class TestEngine:
    """执行引擎基础测试。"""

    def test_simple_expression(self):
        result = execute("1 + 2", {})
        assert result["result"] == 3

    def test_variable_assignment(self):
        result = execute("x = 42", {})
        assert result["result"] is None

    def test_last_expression_auto_return(self):
        result = execute("x = 10\nx * 2", {})
        assert result["result"] == 20

    def test_explicit_result_variable(self):
        result = execute("x = 10\nresult = x * 3", {})
        assert result["result"] == 30

    def test_stdout_capture(self):
        result = execute("print('hello')", {})
        assert result["stdout"] == "hello\n"

    def test_loop(self):
        result = execute("total = 0\nfor i in range(5):\n    total += i\ntotal", {})
        assert result["result"] == 10

    def test_function_definition(self):
        code = "def double(x):\n    return x * 2\ndouble(21)"
        result = execute(code, {})
        assert result["result"] == 42

    def test_try_except(self):
        code = "try:\n    1/0\nexcept ZeroDivisionError:\n    result = 'caught'"
        result = execute(code, {})
        assert result["result"] == "caught"

    def test_list_comprehension(self):
        result = execute("[x**2 for x in range(5)]", {})
        assert result["result"] == [0, 1, 4, 9, 16]

    def test_fstring(self):
        result = execute("name = 'world'\nf'hello {name}'", {})
        assert result["result"] == "hello world"

    def test_repl_state(self):
        """跨步骤状态保持。"""
        state = {}
        ns = {}
        execute("x = 42", ns, state)
        result = execute("x + 8", ns, state)
        assert result["result"] == 50

    def test_injected_function(self):
        """调用注入的函数。"""
        def greet(name):
            return f"hello {name}"
        result = execute("greet('world')", {"greet": greet})
        assert result["result"] == "hello world"


# ============================================================
# 安全边界测试
# ============================================================

class TestSecurity:
    """安全边界测试 — 所有逃逸路径应被拦截。"""

    def test_import_blocked(self):
        result = execute("import os", {})
        assert "error" in result
        assert "SyntaxError" in result["error"]

    def test_from_import_blocked(self):
        result = execute("from os import path", {})
        assert "error" in result
        assert "SyntaxError" in result["error"]

    def test_class_allowed(self):
        result = execute("class Foo:\n    x = 1\nFoo.x", {})
        assert result["result"] == 1

    def test_eval_not_available(self):
        result = execute("eval('1+1')", {})
        assert "error" in result
        assert "NameError" in result["error"]

    def test_exec_not_available(self):
        result = execute("exec('x=1')", {})
        assert "error" in result
        assert "NameError" in result["error"]

    def test_open_not_available(self):
        result = execute("open('file.txt')", {})
        assert "error" in result
        assert "NameError" in result["error"]

    def test_getattr_available(self):
        result = execute("getattr([], 'append')", {})
        assert result["result"] is not None

    def test_globals_available(self):
        result = execute("type(globals())", {})
        assert result["result"] == dict

    def test_dir_available(self):
        result = execute("'append' in dir([])", {})
        assert result["result"] is True

    def test_dunder_import_not_available(self):
        result = execute("__import__('os')", {})
        assert "error" in result
        assert "NameError" in result["error"]

    def test_compile_not_available(self):
        result = execute("compile('x', '', 'exec')", {})
        assert "error" in result
        assert "NameError" in result["error"]

    def test_type_single_arg_ok(self):
        result = execute("type(42)", {})
        assert result["result"] == int

    def test_type_three_args_allowed(self):
        result = execute("type('X', (object,), {'x': 1})", {})
        assert result["result"] is not None
        assert result["result"].x == 1

    def test_safe_builtins_available(self):
        """安全内置函数应该可用。"""
        result = execute("len([1,2,3])", {})
        assert result["result"] == 3

        result = execute("sorted([3,1,2])", {})
        assert result["result"] == [1, 2, 3]

        result = execute("str(42)", {})
        assert result["result"] == "42"

    def test_traceback_filters_internal_frames(self):
        """traceback 只包含 <pysandbox> 帧，不泄露引擎路径。"""
        result = execute("x = 1\ny = x / 0", {})
        assert "error" in result
        assert "ZeroDivisionError" in result["error"]
        tb = result.get("traceback", "")
        assert "_engine.py" not in tb
        assert "<pysandbox>" in tb

    def test_traceback_no_internal_paths(self):
        """嵌套调用的 traceback 也不应泄露内部路径。"""
        code = "def foo():\n    raise ValueError('bad')\nfoo()"
        result = execute(code, {})
        assert "error" in result
        tb = result.get("traceback", "")
        assert "_engine.py" not in tb
        assert "_app_impl.py" not in tb

    def test_traceback_injected_function_no_leak(self):
        """注入函数抛错时，traceback 不泄露外部路径。"""
        def bad_fn():
            raise RuntimeError("boom")

        result = execute("bad_fn()", {"bad_fn": bad_fn})
        assert "error" in result
        assert "RuntimeError" in result["error"]
        tb = result.get("traceback", "")
        # 只保留 <pysandbox> 帧，外部调用栈不泄露
        assert "test_sandbox.py" not in tb
        assert "_engine.py" not in tb
        assert "<pysandbox>" in tb
        assert "_app_impl.py" not in tb


# ============================================================
# 命名空间测试
# ============================================================

class TestNamespace:
    """命名空间机制测试。"""

    def test_register_and_access(self):
        ns = Namespace("test")
        ns.register("greet", lambda name: f"hi {name}", "Greet someone")
        assert ns.greet("world") == "hi world"

    def test_unknown_attr_raises(self):
        ns = Namespace("test")
        with pytest.raises(AttributeError):
            ns.nonexistent()

    def test_repr(self):
        ns = Namespace("browser")
        ns.register("click", lambda: None)
        assert "browser" in repr(ns)
        assert "1 functions" in repr(ns)


class TestNamespaceRegistry:
    """命名空间注册表测试。"""

    def test_build_namespace_dict(self):
        registry = NamespaceRegistry()
        ns = Namespace("demo")
        ns.register("ping", lambda: "pong", "Ping")
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        assert "demo" in ns_dict
        assert "help" in ns_dict
        assert "list_functions" not in ns_dict  # list_functions 已移除

    def test_help_function(self):
        registry = NamespaceRegistry()
        ns = Namespace("tools")

        def my_func():
            """Detailed documentation here."""
            pass

        ns.register("my_func", my_func, "Short desc")
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        help_fn = ns_dict["help"]
        output = help_fn(ns.my_func)
        assert "Detailed documentation" in output

    def test_help_namespace(self):
        registry = NamespaceRegistry()
        ns = Namespace("demo")
        ns.register("a", lambda: None, "func a")
        ns.register("b", lambda: None, "func b")
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        output = ns_dict["help"](ns)
        assert "demo" in output
        assert "func a" in output

    def test_help_string_lookup(self):
        registry = NamespaceRegistry()
        ns = Namespace("tools")

        def my_func():
            """Documentation via string lookup."""
            pass

        ns.register("my_func", my_func)
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        output = ns_dict["help"]("tools.my_func")
        assert "Documentation via string lookup" in output

    def test_namespace_in_sandbox(self):
        """命名空间对象在 sandbox 执行引擎中可用。"""
        registry = NamespaceRegistry()
        ns = Namespace("math")
        ns.register("add", lambda a, b: a + b, "Add")
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        result = execute("math.add(3, 4)", ns_dict)
        assert result["result"] == 7

    def test_help_in_sandbox(self):
        """help() 在 sandbox 中可用。"""
        registry = NamespaceRegistry()
        ns = Namespace("demo")

        def ping():
            """Ping function documentation."""
            return "pong"

        ns.register("ping", ping)
        registry.add(ns)

        ns_dict = registry.build_namespace_dict()
        result = execute("help(demo.ping)", ns_dict)
        assert "Ping function documentation" in result["result"]

    def test_remove_namespace(self):
        registry = NamespaceRegistry()
        ns = Namespace("temp")
        ns.register("foo", lambda: None)
        registry.add(ns)
        registry.remove("temp")

        ns_dict = registry.build_namespace_dict()
        assert "temp" not in ns_dict
