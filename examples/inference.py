"""Read the example YAML and call the PointDINO inference API."""

from __future__ import annotations

import json

from pointdino import PointDINOPredictor
from pointdino.inference import list_images
from tools.config_loader import load_config, resolve_config_path


def main() -> None:
    # YAML loading belongs to the application/example layer.  The reusable
    # predictor only receives the already loaded model configuration.
    config_path = resolve_config_path("config/inference.yaml")
    config = load_config(config_path)
    inference_config = config.inference

    predictor = PointDINOPredictor(
        config,
        inference_config.checkpoint,
        score_threshold=inference_config.score_threshold,
        device=inference_config.device,
        log_level=config.get("log_level", "INFO"),
        log_file=inference_config.get("log_file"),
    )

    images = list_images(inference_config.input)
    if not images:
        raise FileNotFoundError(
            f"no supported images found in: {inference_config.input}")

    # The example runs the first configured image. Use tools.predict for the
    # config-driven batch entry point.
    prediction = predictor.predict_path(images[0])
    print(json.dumps(prediction.to_dict(include_metadata=True), indent=2))


if __name__ == "__main__":
    main()
