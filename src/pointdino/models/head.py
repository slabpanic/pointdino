"""PointDINO head registration."""

from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.registry import MODELS


@MODELS.register_module(name="PointDINOHead")
class PointDINOHead(DINOHead):
    """Classification plus 2D point regression head."""

