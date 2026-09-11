# Copyright (c) OpenMMLab. All rights reserved.
"""Lazy public API imports."""

from importlib import import_module


_INFERENCE_NAMES = {
    'async_inference_detector', 'inference_detector', 'inference_mot',
    'init_detector', 'init_track_model'
}


def __getattr__(name):
    if name in _INFERENCE_NAMES:
        module = import_module('.inference', __name__)
        value = getattr(module, name)
    elif name == 'DetInferencer':
        module = import_module('.det_inferencer', __name__)
        value = module.DetInferencer
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = sorted((*_INFERENCE_NAMES, 'DetInferencer'))
