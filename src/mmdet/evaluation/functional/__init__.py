"""Lazy evaluation helpers.

The full evaluation stack contains optional dataset and plotting dependencies.
Only class names and box overlap helpers are needed by detector inference;
other helpers are imported when a training/evaluation caller asks for them.
"""

from importlib import import_module

from .bbox_overlaps import bbox_overlaps
from .class_names import (cityscapes_classes, coco_classes,
                          coco_panoptic_classes, dataset_aliases, get_classes,
                          imagenet_det_classes, imagenet_vid_classes,
                          objects365v1_classes, objects365v2_classes,
                          oid_challenge_classes, oid_v6_classes, voc_classes)


_LAZY_EXPORTS = {
    'evaluateImgLists': ('.cityscapes_utils', 'evaluateImgLists'),
    'average_precision': ('.mean_ap', 'average_precision'),
    'eval_map': ('.mean_ap', 'eval_map'),
    'print_map_summary': ('.mean_ap', 'print_map_summary'),
    'eval_recalls': ('.recall', 'eval_recalls'),
    'plot_iou_recall': ('.recall', 'plot_iou_recall'),
    'plot_num_recall': ('.recall', 'plot_num_recall'),
    'print_recall_summary': ('.recall', 'print_recall_summary'),
    'INSTANCE_OFFSET': ('.panoptic_utils', 'INSTANCE_OFFSET'),
    'pq_compute_single_core': ('.panoptic_utils', 'pq_compute_single_core'),
    'pq_compute_multi_core': ('.panoptic_utils', 'pq_compute_multi_core'),
    'YTVIS': ('.ytvis', 'YTVIS'),
    'YTVISeval': ('.ytviseval', 'YTVISeval'),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    'voc_classes', 'imagenet_det_classes', 'imagenet_vid_classes',
    'coco_classes', 'cityscapes_classes', 'dataset_aliases',
    'get_classes', 'average_precision', 'eval_map', 'print_map_summary',
    'eval_recalls', 'print_recall_summary', 'plot_num_recall',
    'plot_iou_recall', 'oid_v6_classes', 'oid_challenge_classes',
    'INSTANCE_OFFSET', 'pq_compute_single_core', 'pq_compute_multi_core',
    'bbox_overlaps', 'objects365v1_classes', 'objects365v2_classes',
    'coco_panoptic_classes', 'evaluateImgLists', 'YTVIS', 'YTVISeval'
]
