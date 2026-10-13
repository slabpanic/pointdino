# mmcv build compatibility

This package is injected only into uv's isolated build environment for mmcv
2.2.0. It patches the upstream `setup.py` version loader for Python 3.13+ and
rejects the build when the CUDA Toolkit or `nvcc` cannot be detected. The
NVIDIA driver and PyTorch CUDA runtime wheels are not sufficient to compile
MMCV extensions. Install a CUDA Toolkit matching `torch.version.cuda`, set
`CUDA_HOME` when needed, and rerun `uv sync`. This package is not installed as
a PointDINO runtime dependency.
