"""Production entry points for the PointDINO point detector."""

__all__ = ["PointDINOPredictor"]


def __getattr__(name):
    if name == "PointDINOPredictor":
        from .inference import PointDINOPredictor

        return PointDINOPredictor
    raise AttributeError(name)
