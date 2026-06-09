"""Tests for mutagent Declaration base class (via mutobj)."""

import pytest
import mutagent
import mutobj
from mutobj import Declaration


class TestDeclarationMeta:

    def test_subclass_is_declaration(self):
        class MyClass(mutobj.Declaration):
            pass

        assert issubclass(MyClass, mutobj.Declaration)


class TestMutagentDeclaration:

    def test_is_mutobj_declaration(self):
        assert mutobj.Declaration is Declaration

    def test_attribute_declaration(self):
        class Item(mutobj.Declaration):
            name: str
            value: int

        item = Item(name="test", value=42)
        assert item.name == "test"
        assert item.value == 42

    def test_attribute_not_set_raises(self):
        class Thing(mutobj.Declaration):
            data: str

        with pytest.raises(TypeError, match=r"missing field.*'data'"):
            Thing()

    def test_stub_method_recognized(self):
        class Service(mutobj.Declaration):
            def process(self) -> str: ...

        assert mutobj.get_declaration_func(Service, "process") is not None

    def test_stub_method_is_default_impl(self):
        class Handler(mutobj.Declaration):
            def handle(self) -> None: ...

        h = Handler()
        # In mutobj, the original function body is kept as default impl
        # A `...` body executes and returns None
        assert h.handle() is None

    def test_impl_works(self):
        class Greeter(mutobj.Declaration):
            name: str

            def greet(self) -> str: ...

        @mutobj.impl(Greeter.greet)
        def greet(self: Greeter) -> str:
            return f"Hello, {self.name}!"

        g = Greeter(name="World")
        assert g.greet() == "Hello, World!"

    def test_impl_override(self):
        class Calc(mutobj.Declaration):
            def compute(self, x: int) -> int: ...

        @mutobj.impl(Calc.compute)
        def compute_v1(self, x: int) -> int:
            return x + 1

        c = Calc()
        assert c.compute(5) == 6

        # In mutobj, later registrations automatically become the active impl
        @mutobj.impl(Calc.compute)
        def compute_v2(self, x: int) -> int:
            return x * 2

        assert c.compute(5) == 10

    def test_isinstance_check(self):
        class Agent(mutobj.Declaration):
            pass

        a = Agent()
        assert isinstance(a, Agent)
        assert isinstance(a, mutobj.Declaration)
        assert isinstance(a, Declaration)
