"""mutagent.builtins.patch_module -- patch_module tool implementation."""

import mutagent
from mutagent.essential_tools import EssentialTools


@mutagent.impl(EssentialTools.patch_module)
def patch_module(self: EssentialTools, module_path: str, source: str) -> str:
    """Patch a module with new source code at runtime."""
    try:
        mod = self.module_manager.patch_module(module_path, source)
        version = self.module_manager.get_version(module_path)
        return f"OK: {module_path} patched (v{version})"
    except Exception as e:
        return f"Error patching {module_path}: {type(e).__name__}: {e}"
