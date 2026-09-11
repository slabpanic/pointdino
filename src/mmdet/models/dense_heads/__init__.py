"""Dense-head modules are imported on demand."""

from importlib import import_module
import pkgutil


def __getattr__(name):
    for module_info in pkgutil.iter_modules(__path__):
        module = import_module(f'{__name__}.{module_info.name}')
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(name)
