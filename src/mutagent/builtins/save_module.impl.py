"""mutagent.builtins.save_module -- save_module tool implementation."""

import mutagent
from mutagent.essential_tools import EssentialTools


@mutagent.impl(EssentialTools.save_module)
def save_module(self: EssentialTools, module_path: str, file_path: str = "") -> str:
    """Persist a patched module to the filesystem."""
    try:
        directory = file_path if file_path else "."
        path = self.module_manager.save_module(module_path, directory)
        return f"OK: {module_path} saved to {path}"
    except Exception as e:
        return f"Error saving {module_path}: {type(e).__name__}: {e}"
