"""Run batch inference from the repository YAML configuration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from pointdino import PointDINOPredictor
from pointdino.inference import list_images
from pointdino.logging_utils import logger
from tools.config_loader import default_config_path, load_config, resolve_config_path


DEFAULT_CONFIG = default_config_path("inference.yaml")


def _draw_points(image, points, scores, threshold):
    for (x, y), score in zip(points, scores):
        if score < threshold:
            continue
        cv2.circle(image, (round(x), round(y)), 4, (0, 0, 255), -1)
    return image


def predict(config_path: str | Path = DEFAULT_CONFIG) -> None:
    """Run the configured batch inference job."""
    config_path = resolve_config_path(config_path)
    cfg = load_config(config_path)
    inference_cfg = cfg.get("inference", {})
    checkpoint = inference_cfg.get("checkpoint")
    input_path = inference_cfg.get("input")
    output_path = inference_cfg.get("output")
    visualization_dir = inference_cfg.get("visualization_dir")
    if not checkpoint:
        raise ValueError("inference.checkpoint must be set in config/inference.yaml")
    if not input_path:
        raise ValueError("inference.input must be set in config/inference.yaml")
    if not output_path:
        raise ValueError("inference.output must be set in config/inference.yaml")

    score_threshold = float(inference_cfg.get("score_threshold", 0.5))
    predictor = PointDINOPredictor(
        cfg,
        checkpoint,
        score_threshold=score_threshold,
        device=inference_cfg.get("device"),
        log_level=cfg.get("log_level", "INFO"),
        log_file=inference_cfg.get("log_file"),
    )

    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"no supported images found in: {input_path}")

    logger.info("Processing {} image(s)", len(images))
    results = {}
    for image_path in images:
        prediction = predictor.predict_path(image_path)
        results[str(image_path)] = prediction.to_dict(include_metadata=True)
        if visualization_dir:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode image: {image_path}")
            vis = _draw_points(
                image, prediction.points, prediction.scores, score_threshold)
            vis_path = Path(visualization_dir) / image_path.name
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(vis_path), vis):
                raise OSError(f"failed to write visualization: {vis_path}")

    payload = next(iter(results.values())) if len(results) == 1 else results
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if inference_cfg.get("pretty", False) else None,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text + "\n", encoding="utf-8")
    logger.info("Prediction JSON written to {}", output_file)


if __name__ == "__main__":
    predict()
