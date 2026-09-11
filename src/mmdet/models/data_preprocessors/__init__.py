"""Lazy data-preprocessor exports."""

from importlib import import_module


_MODULES = {
    "BatchFixedSizePad": ".data_preprocessor",
    "BatchResize": ".data_preprocessor",
    "BatchSyncRandomResize": ".data_preprocessor",
    "BoxInstDataPreprocessor": ".data_preprocessor",
    "DetDataPreprocessor": ".data_preprocessor",
    "MultiBranchDataPreprocessor": ".data_preprocessor",
    "ReIDDataPreprocessor": ".reid_data_preprocessor",
    "TrackDataPreprocessor": ".track_data_preprocessor",
}

__all__ = tuple(_MODULES)


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
