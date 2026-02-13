"""mutagent base module - Object base class and MutagentMeta metaclass."""

from __future__ import annotations

from typing import Any

from forwardpy import Object as _ForwardpyObject
from forwardpy.core import ObjectMeta


class MutagentMeta(ObjectMeta):
    """mutagent metaclass extending forwardpy.ObjectMeta.

    Adds a class registry that maps (module, qualname) to class objects.
    Phase 2 (Task 2.2) will add in-place class redefinition support.
    """

    _class_registry = {}  # type: dict[tuple[str, str], type]

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> MutagentMeta:
        # Delegate all class creation to forwardpy.ObjectMeta
        cls = super().__new__(mcs, name, bases, namespace)

        # Register the class for future in-place redefinition (Phase 2)
        module = namespace.get("__module__", "")
        qualname = namespace.get("__qualname__", name)
        key = (module, qualname)
        if key not in mcs._class_registry:
            mcs._class_registry[key] = cls

        return cls


class Object(_ForwardpyObject, metaclass=MutagentMeta):
    """mutagent unified base class.

    All mutagent core classes inherit from this. Currently a thin wrapper
    around forwardpy.Object with MutagentMeta as the metaclass.
    """

    pass
