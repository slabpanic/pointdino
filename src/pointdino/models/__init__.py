"""PointDINO model aliases registered under a project-specific namespace."""

from .detector import PointDINO
from .head import PointDINOHead

__all__ = ["PointDINO", "PointDINOHead"]


def register_inference_modules() -> None:
    """Register only modules used by the PointDINO inference graph.

    The full MMDetection module walk is intentionally reserved for training.
    Keeping this list explicit prevents optional datasets, evaluators and
    visualizers from becoming import-time inference dependencies.
    """
    from mmdet.datasets.transforms import formatting as _formatting  # noqa: F401
    from mmdet.datasets.transforms import loading as _loading  # noqa: F401
    from mmdet.models.backbones import resnet as _resnet  # noqa: F401
    from mmdet.models.data_preprocessors import data_preprocessor as _preprocessor  # noqa: F401,E501
    from mmdet.models.losses import focal_loss as _focal_loss  # noqa: F401
    from mmdet.models.losses import iou_loss as _iou_loss  # noqa: F401
    from mmdet.models.losses import smooth_l1_loss as _smooth_l1_loss  # noqa: F401,E501
    from mmdet.models.necks import channel_mapper as _channel_mapper  # noqa: F401
