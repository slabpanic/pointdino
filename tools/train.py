"""Run training from the repository YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

from pointdino import models as _models  # noqa: F401
from pointdino import evaluation as _evaluation  # noqa: F401
from pointdino.logging_utils import configure_logging, logger
from tools.config_loader import default_config_path, load_config, resolve_config_path


DEFAULT_CONFIG = default_config_path("train.yaml")


def train(config_path: str | Path = DEFAULT_CONFIG) -> Any:
    """Build and run the runner configured in the repository YAML."""
    config_path = resolve_config_path(config_path)
    cfg = load_config(config_path)

    if cfg.get("amp"):
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    log_file = cfg.get("log_file") or str(Path(cfg.work_dir) / "pointdino.log")
    configure_logging(cfg.get("log_level", "INFO"), log_file)
    register_all_modules(init_default_scope=True)
    logger.info("Starting PointDINO training")
    logger.info("Config: {}", config_path)
    logger.info("Work directory: {}", cfg.work_dir)

    runner = (RUNNERS.build(cfg) if "runner_type" in cfg else
              Runner.from_cfg(cfg))
    result = runner.train()
    logger.info("PointDINO training finished")
    return result


if __name__ == "__main__":
    train()
