"""Patch mmcv 2.2.0 for Python 3.13+ and validate CUDA builds.

Python 3.13 changed the visibility of names created by ``exec`` in a function's
local scope.  mmcv 2.2.0 still relies on that old behavior in ``setup.py``.
This module is installed only as an extra uv build dependency for mmcv, so the
patch applies to the temporary source tree used for building mmcv and is not
installed into PointDINO's runtime environment.
"""

import os
from pathlib import Path
import shutil


def _cuda_toolkit_home() -> Path | None:
    """Find a CUDA Toolkit matching the Torch CUDA major/minor version."""

    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        home = Path(configured)
        return home if (home / "bin" / "nvcc").is_file() else None

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent

    candidates = [Path("/usr/local/cuda")]
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*")))
    candidates = [
        candidate for candidate in candidates
        if (candidate / "bin" / "nvcc").is_file()
    ]
    if not candidates:
        return None

    try:
        import torch
        expected = f"cuda-{torch.version.cuda}"
    except (ImportError, AttributeError):
        expected = None

    for candidate in candidates:
        if candidate.name == expected:
            return candidate
    return candidates[0]


def _cuda_toolkit_available() -> bool:
    """Return whether nvcc is available to the isolated build.

    A CUDA-enabled PyTorch wheel can report ``torch.cuda.is_available()`` as
    true when only the NVIDIA driver is installed.  MMCV needs the CUDA
    Toolkit, however, because its CUDA extensions are compiled with nvcc.
    """

    cuda_home = _cuda_toolkit_home()
    if cuda_home is None:
        return False

    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    bin_path = str(cuda_home / "bin")
    if bin_path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    return True


def _require_cuda_toolkit() -> None:
    """Record a clear build error when the CUDA Toolkit is unavailable.

    The NVIDIA driver alone is not enough to compile MMCV's CUDA extensions;
    the isolated build must also be able to find ``nvcc``.
    """

    if not _cuda_toolkit_available():
        os.environ["MMCV_CUDA_TOOLKIT_MISSING"] = "1"


def _patch_mmcv_setup() -> None:
    setup_path = Path.cwd() / "setup.py"
    if not setup_path.is_file():
        return

    source = setup_path.read_text(encoding="utf-8")
    old = """    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
    new = """    namespace = {}
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), namespace)
    return namespace['__version__']"""

    old_extensions = """def get_extensions():
    extensions = []"""
    new_extensions = """def get_extensions():
    if os.getenv('MMCV_CUDA_TOOLKIT_MISSING') == '1':
        raise RuntimeError(
            'MMCV CUDA build requires the CUDA Toolkit and nvcc. '
            'Install a CUDA Toolkit matching torch.version.cuda and set '
            'CUDA_HOME, then rerun uv sync.'
        )
    extensions = []"""

    if old in source:
        source = source.replace(old, new)
    if old_extensions in source:
        source = source.replace(old_extensions, new_extensions, 1)
    setup_path.write_text(source, encoding="utf-8")


_require_cuda_toolkit()
_patch_mmcv_setup()
