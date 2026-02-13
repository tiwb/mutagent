"""mutagent base module - Object base class and MutagentMeta metaclass."""

from __future__ import annotations

from typing import Any

from forwardpy import Object as _ForwardpyObject
from forwardpy.core import (
    ObjectMeta,
    ForwardpyProperty,
    _method_registry,
    _attribute_registry,
    _property_registry,
)


def _update_class_inplace(existing: type, new_cls: type) -> None:
    """Update an existing class in-place with attributes from a new definition.

    Transplants user-defined attributes from new_cls onto existing,
    removes attributes that were deleted in the new definition,
    and fixes forwardpy internal references.
    """
    _SKIP = frozenset({
        "__dict__", "__weakref__", "__class__", "__mro__",
        "__subclasses__", "__bases__", "__name__", "__qualname__",
        "__module__",
    })

    old_attrs = set(existing.__dict__.keys())
    new_attrs = set(new_cls.__dict__.keys())

    # Delete attributes removed in new definition
    for attr in old_attrs - new_attrs - _SKIP:
        try:
            delattr(existing, attr)
        except (AttributeError, TypeError):
            pass

    # Set/update attributes from new definition
    for attr in new_attrs - _SKIP:
        val = new_cls.__dict__[attr]
        try:
            setattr(existing, attr, val)
        except (AttributeError, TypeError):
            pass

        # Fix __forwardpy_class__ references on stubs/impls
        if callable(val) and hasattr(val, "__forwardpy_class__"):
            val.__forwardpy_class__ = existing
        # Handle classmethod/staticmethod wrappers
        if isinstance(val, (classmethod, staticmethod)):
            inner = val.__func__
            if hasattr(inner, "__forwardpy_class__"):
                inner.__forwardpy_class__ = existing
        # Fix ForwardpyProperty.owner_cls
        if isinstance(val, ForwardpyProperty):
            val.owner_cls = existing

    # Update annotations
    if "__annotations__" in new_cls.__dict__:
        existing.__annotations__ = dict(new_cls.__dict__["__annotations__"])


def _migrate_forwardpy_registries(existing: type, new_cls: type) -> None:
    """Move forwardpy registry entries from new_cls to existing.

    For method registry: merge (preserving existing impls that new_cls didn't add).
    Then re-apply all registered impls onto the class to overwrite any stubs
    that _update_class_inplace transplanted from new_cls.
    """
    from forwardpy.core import _DECLARED_CLASSMETHODS, _DECLARED_STATICMETHODS

    # Merge method registries (existing impls survive if not overridden by new_cls)
    if new_cls in _method_registry:
        new_methods = _method_registry.pop(new_cls)
        if existing not in _method_registry:
            _method_registry[existing] = {}
        _method_registry[existing].update(new_methods)

    # Re-apply all registered impls back onto the class
    # (stubs from new_cls may have overwritten them during _update_class_inplace)
    declared_cm = getattr(existing, _DECLARED_CLASSMETHODS, set())
    declared_sm = getattr(existing, _DECLARED_STATICMETHODS, set())
    for method_name, impl_func in _method_registry.get(existing, {}).items():
        if method_name in declared_cm:
            setattr(existing, method_name, classmethod(impl_func))
        elif method_name in declared_sm:
            setattr(existing, method_name, staticmethod(impl_func))
        else:
            setattr(existing, method_name, impl_func)

    # Attribute registry: replace (new definition is authoritative)
    if new_cls in _attribute_registry:
        _attribute_registry[existing] = _attribute_registry.pop(new_cls)

    # Property registry: replace, fixing owner_cls references
    if new_cls in _property_registry:
        props = _property_registry.pop(new_cls)
        for prop in props.values():
            prop.owner_cls = existing
        _property_registry[existing] = props


class MutagentMeta(ObjectMeta):
    """mutagent metaclass extending forwardpy.ObjectMeta.

    Supports in-place class redefinition: when a class with the same
    (module, qualname) key is created again, the existing class object
    is updated in-place rather than creating a new one. This preserves
    class identity (id(cls)), isinstance checks, and @impl registrations.
    """

    _class_registry = {}  # type: dict[tuple[str, str], type]

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> MutagentMeta:
        module = namespace.get("__module__", "")
        qualname = namespace.get("__qualname__", name)
        key = (module, qualname)

        existing = mcs._class_registry.get(key)

        # Always create via super() -- ObjectMeta needs to process
        # stubs, attribute descriptors, property registries, etc.
        new_cls = super().__new__(mcs, name, bases, namespace)

        if existing is not None and existing is not new_cls:
            # In-place update: transplant new definition onto existing class
            _update_class_inplace(existing, new_cls)
            _migrate_forwardpy_registries(existing, new_cls)
            return existing

        # First definition -- register it
        mcs._class_registry[key] = new_cls
        return new_cls


class Object(_ForwardpyObject, metaclass=MutagentMeta):
    """mutagent unified base class.

    All mutagent core classes inherit from this. Currently a thin wrapper
    around forwardpy.Object with MutagentMeta as the metaclass.
    """

    pass
