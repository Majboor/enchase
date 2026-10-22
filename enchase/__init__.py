"""enCHASE — alias of the :mod:`chase` package.

``import enchase`` gives exactly the ``chase`` module (same functions, same
submodules), so code can use the distribution name or the short import name.
"""

import importlib
import importlib.abc
import importlib.util
import sys

import chase as _chase


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``enchase.<x>`` to the already-importable ``chase.<x>``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("enchase."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("chase." + spec.name[len("enchase."):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = _chase
