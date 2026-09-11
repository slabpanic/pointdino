# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import importlib
import platform
import pkgutil
import warnings

import cv2
import torch.multiprocessing as mp
from mmengine import DefaultScope


def setup_cache_size_limit_of_dynamo():
    """Keep runtime settings in YAML instead of reading the environment."""
    return


def setup_multi_processes(cfg):
    """Apply multiprocessing and OpenCV settings from the YAML config."""
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f' It will be force set to `{mp_start_method}`. Configure it '
                'in the YAML file if needed.')
        mp.set_start_method(mp_start_method, force=True)

    cv2.setNumThreads(cfg.get('opencv_num_threads', 0))


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmdet into the registries."""
    for package_name in (
            'mmdet.datasets', 'mmdet.engine', 'mmdet.evaluation',
            'mmdet.models', 'mmdet.visualization'):
        package = importlib.import_module(package_name)
        if not hasattr(package, '__path__'):
            continue
        for module_info in pkgutil.walk_packages(
                package.__path__, package.__name__ + '.'):
            importlib.import_module(module_info.name)

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmdet')
        if never_created:
            DefaultScope.get_instance('mmdet', scope_name='mmdet')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmdet':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmdet", '
                          '`register_all_modules` will force the current '
                          'default scope to be "mmdet". If this is not '
                          'expected, please set `init_default_scope=False`.')
            new_instance_name = f'mmdet-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmdet')
