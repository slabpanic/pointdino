"""Production entry points for the PointDINO point detector."""

from ._compat import patch_python314_compat

patch_python314_compat()

__all__ = ["PointDINOPredictor"]


def __getattr__(name):
    if name == "PointDINOPredictor":
        from .inference import PointDINOPredictor

        return PointDINOPredictor
    raise AttributeError(name)
