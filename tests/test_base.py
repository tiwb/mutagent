"""Tests for mutagent.Object base class and MutagentMeta."""

import pytest
import mutagent
from mutagent.base import MutagentMeta
from forwardpy.core import ObjectMeta, _DECLARED_METHODS


class TestMutagentMeta:

    def test_inherits_from_objectmeta(self):
        assert issubclass(MutagentMeta, ObjectMeta)

    def test_object_uses_mutagent_meta(self):
        assert type(mutagent.Object) is MutagentMeta

    def test_subclass_uses_mutagent_meta(self):
        class MyClass(mutagent.Object):
            pass

        assert type(MyClass) is MutagentMeta

    def test_class_registered_in_registry(self):
        class RegTest(mutagent.Object):
            pass

        key = (RegTest.__module__, RegTest.__qualname__)
        assert key in MutagentMeta._class_registry
        assert MutagentMeta._class_registry[key] is RegTest

    def test_mutagent_object_registered(self):
        key = ("mutagent.base", "Object")
        assert key in MutagentMeta._class_registry


class TestMutagentObject:

    def test_inherits_from_forwardpy_object(self):
        import forwardpy

        assert issubclass(mutagent.Object, forwardpy.Object)

    def test_attribute_declaration(self):
        class Item(mutagent.Object):
            name: str
            value: int

        item = Item(name="test", value=42)
        assert item.name == "test"
        assert item.value == 42

    def test_attribute_not_set_raises(self):
        class Thing(mutagent.Object):
            data: str

        t = Thing()
        with pytest.raises(AttributeError):
            _ = t.data

    def test_stub_method_recognized(self):
        class Service(mutagent.Object):
            def process(self) -> str: ...

        declared = getattr(Service, _DECLARED_METHODS, set())
        assert "process" in declared

    def test_stub_method_raises_not_implemented(self):
        class Handler(mutagent.Object):
            def handle(self) -> None: ...

        h = Handler()
        with pytest.raises(NotImplementedError):
            h.handle()

    def test_impl_works(self):
        class Greeter(mutagent.Object):
            name: str

            def greet(self) -> str: ...

        @mutagent.impl(Greeter.greet)
        def greet(self: Greeter) -> str:
            return f"Hello, {self.name}!"

        g = Greeter(name="World")
        assert g.greet() == "Hello, World!"

    def test_impl_override(self):
        class Calc(mutagent.Object):
            def compute(self, x: int) -> int: ...

        @mutagent.impl(Calc.compute)
        def compute_v1(self, x: int) -> int:
            return x + 1

        c = Calc()
        assert c.compute(5) == 6

        @mutagent.impl(Calc.compute, override=True)
        def compute_v2(self, x: int) -> int:
            return x * 2

        assert c.compute(5) == 10

    def test_isinstance_check(self):
        import forwardpy

        class Agent(mutagent.Object):
            pass

        a = Agent()
        assert isinstance(a, Agent)
        assert isinstance(a, mutagent.Object)
        assert isinstance(a, forwardpy.Object)
