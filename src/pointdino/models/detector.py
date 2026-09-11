"""PointDINO detector registration.

The implementation lives in the local MMDetection fork because it reuses the
deformable-attention runtime. This alias gives the extracted project a stable,
unambiguous model name and keeps configs decoupled from the generic ``DINO``
name.
"""

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS


@MODELS.register_module(name="PointDINO")
class PointDINO(DINO):
    """Query-based single-class point detector."""

