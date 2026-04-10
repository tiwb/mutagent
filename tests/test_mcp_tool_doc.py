"""测试 MCPToolProvider.list_tools 使用声明 docstring 而非 @impl docstring。"""

import mutobj
from mutagent.net._mcp_impl import _get_declaration_doc


class TestGetDeclarationDoc:
    """测试 _get_declaration_doc 辅助函数"""

    def test_returns_declaration_docstring(self):
        """声明方法有 docstring 时，能正确取回"""
        class Svc(mutobj.Declaration):
            def run(self) -> str:
                """Run the service."""
                ...

        assert _get_declaration_doc(Svc, "run") == "Run the service."

    def test_returns_none_for_no_docstring(self):
        """声明方法没有 docstring 时，返回 None"""
        class Svc2(mutobj.Declaration):
            def run(self) -> str: ...

        assert _get_declaration_doc(Svc2, "run") is None

    def test_returns_none_for_nonexistent_method(self):
        """查询不存在的方法时，返回 None"""
        class Svc3(mutobj.Declaration):
            def run(self) -> str:
                """Run."""
                ...

        assert _get_declaration_doc(Svc3, "no_such_method") is None

    def test_returns_original_after_impl_override(self):
        """@impl 覆盖后，仍能取回原始声明的 docstring"""
        class Svc4(mutobj.Declaration):
            def greet(self) -> str:
                """Original greeting doc."""
                ...

        def greet_impl(self) -> str:
            """Overridden doc."""
            return "hello"
        greet_impl.__module__ = "test_mcp_doc_impl"
        mutobj.impl(Svc4.greet)(greet_impl)

        try:
            # @impl 覆盖了类方法的 __doc__
            assert Svc4.greet.__doc__ == "Overridden doc."
            # _get_declaration_doc 取回的是原始声明的 docstring
            assert _get_declaration_doc(Svc4, "greet") == "Original greeting doc."
        finally:
            mutobj.unregister_module_impls("test_mcp_doc_impl")

    def test_returns_none_for_non_declaration_class(self):
        """对非 Declaration 类查询，返回 None"""
        class Plain:
            def run(self): ...

        assert _get_declaration_doc(Plain, "run") is None

    def test_traverses_mro(self):
        """通过 MRO 查找父类的声明 docstring"""
        class Base(mutobj.Declaration):
            def action(self) -> str:
                """Base action doc."""
                ...

        class Child(Base):
            pass

        assert _get_declaration_doc(Child, "action") == "Base action doc."
