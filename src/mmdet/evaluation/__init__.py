"""Lazy evaluation namespace."""

from .functional import get_classes


def __getattr__(name):
    if name == 'INSTANCE_OFFSET':
        from .functional import INSTANCE_OFFSET

        globals()[name] = INSTANCE_OFFSET
        return INSTANCE_OFFSET
    raise AttributeError(name)


__all__ = ['get_classes', 'INSTANCE_OFFSET']
