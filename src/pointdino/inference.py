"""Reusable PointDINO inference API.

The class delegates image loading, resize and normalization to the MMDetection
test pipeline from the supplied model configuration. This keeps inference
coordinates consistent with training for both native-resolution and resized
datasets. Configuration-file loading is intentionally left to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging_utils import configure_logging, logger


# ``inference.py`` lives in ``src/pointdino`` in the source tree. Relative
# checkpoint and image paths are resolved from the caller's working directory
# first, then from the repository root for source-tree usage.
_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (_PACKAGE_PARENT.parent
                if _PACKAGE_PARENT.name == "src" else _PACKAGE_PARENT)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_path(path: str | Path, *, kind: str) -> Path:
    value = Path(path).expanduser()
    candidates = ([value] if value.is_absolute() else
                  [Path.cwd() / value, PROJECT_ROOT / value])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{kind} does not exist; searched: {searched}")


@dataclass(frozen=True)
class PointPrediction:
    """JSON-friendly output for one image."""

    points: list[list[float]]
    scores: list[float]
    labels: list[int]
    image_size: tuple[int, int]

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "points": self.points,
            "scores": self.scores,
        }
        if include_metadata:
            result.update({
                "labels": self.labels,
                "image_size": {
                    "height": self.image_size[0],
                    "width": self.image_size[1],
                },
            })
        return result


class PointDINOPredictor:
    """Load a PointDINO checkpoint once and run inference repeatedly."""

    def __init__(
        self,
        model_config: Mapping[str, Any] | Any,
        checkpoint_path: str | Path,
        score_threshold: float = 0.5,
        device: str | None = None,
        *,
        log_level: str = "INFO",
        log_file: str | Path | None = None,
    ) -> None:
        if not 0.0 <= float(score_threshold) <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")

        import torch
        from mmengine.config import Config
        from mmdet.apis import init_detector
        from . import models as _models
        from .models import register_inference_modules

        checkpoint = _resolve_path(checkpoint_path, kind="checkpoint")
        self.score_threshold = float(score_threshold)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else
                                 "cpu")
        self.checkpoint_path = checkpoint
        if isinstance(model_config, Config):
            model_cfg = deepcopy(model_config)
        elif isinstance(model_config, Mapping):
            model_cfg = Config(deepcopy(dict(model_config)))
        else:
            raise TypeError(
                "model_config must be an mmengine Config or a mapping; "
                f"got {type(model_config)!r}")
        self.config = model_cfg
        configure_logging(log_level, log_file)
        logger.info("Loading PointDINO checkpoint: {}", checkpoint)
        register_inference_modules()
        inference_model_cfg = deepcopy(model_cfg)
        # Hungarian matching is needed for training, not for constructing an
        # inference-only model. Removing it also keeps scipy out of the base
        # inference installation.
        inference_model_cfg.model.train_cfg = None
        self.model = init_detector(
            inference_model_cfg,
            str(checkpoint),
            palette="random",
            device=self.device,
        )
        logger.info("PointDINO model ready on {}", self.device)

    @staticmethod
    def _validate_image(image: Any) -> None:
        import numpy as np

        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image)!r}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape [H, W, 3], got {image.shape}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")

    @staticmethod
    def _prediction_from_sample(sample: Any,
                                image_size: tuple[int, int],
                                threshold: float) -> PointPrediction:
        if not hasattr(sample, "pred_instances"):
            raise RuntimeError("PointDINO output has no pred_instances")
        instances = sample.pred_instances
        points = instances.points.detach().cpu()
        scores = instances.scores.detach().cpu()
        labels = getattr(instances, "labels", None)
        if labels is None:
            labels = scores.new_zeros(scores.shape).long()
        labels = labels.detach().cpu()
        keep = scores >= threshold
        return PointPrediction(
            points=points[keep].tolist(),
            scores=scores[keep].tolist(),
            labels=labels[keep].tolist(),
            image_size=image_size,
        )

    def predict_result(self, image: Any) -> PointPrediction:
        """Predict points from an OpenCV BGR ``H x W x 3`` image."""
        import torch

        self._validate_image(image)
        from mmdet.apis import inference_detector

        with torch.inference_mode():
            sample = inference_detector(self.model, image)
        return self._prediction_from_sample(
            sample, (int(image.shape[0]), int(image.shape[1])),
            self.score_threshold)

    def predict(self, image: Any) -> dict[str, Any]:
        """Return the legacy ``{"points": ..., "scores": ...}`` shape."""
        return self.predict_result(image).to_dict()

    def predict_path(self, image_path: str | Path) -> PointPrediction:
        """Predict points from an image file using the configured pipeline."""
        from mmdet.apis import inference_detector

        path = _resolve_path(image_path, kind="image")
        sample = inference_detector(self.model, str(path))

        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {path}")
        return self._prediction_from_sample(
            sample, (int(image.shape[0]), int(image.shape[1])),
            self.score_threshold)


def list_images(path: str | Path) -> list[Path]:
    """Return sorted image files for the CLI's file-or-directory input."""
    root = _resolve_path(path, kind="input")
    if root.is_file():
        return [root]
    return sorted(
        item for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
