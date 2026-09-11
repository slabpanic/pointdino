"""PointDINO evaluation registration."""

from mmdet.evaluation.metrics.point_metric import PointMetric
from mmdet.registry import METRICS


@METRICS.register_module(name="PointDINOMetric")
class PointDINOMetric(PointMetric):
    """One-to-one pixel-distance metrics for PointDINO predictions."""

