"""Runtime environment checks for PyTorch, Paddle, and NVIDIA dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any


@dataclass()
class EnvironmentStatus:
    """Collected runtime information for the project environment."""

    torch_version: str | None
    torch_cuda_available: bool | None
    paddle_version: str | None
    paddle_cuda_compiled: bool | None
    nvidia_packages: dict[str, str | None]


def _safe_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_status() -> EnvironmentStatus:
    """Collect environment details without performing any installation."""

    torch_version: str | None = None
    torch_cuda_available: bool | None = None
    paddle_version: str | None = None
    paddle_cuda_compiled: bool | None = None

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        import paddle

        paddle_version = paddle.__version__
        paddle_cuda_compiled = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        pass

    nvidia_packages = {
        "nvidia-cudnn-cu12": _safe_version("nvidia-cudnn-cu12"),
        "nvidia-cusparselt-cu12": _safe_version("nvidia-cusparselt-cu12"),
        "nvidia-nccl-cu12": _safe_version("nvidia-nccl-cu12"),
    }

    return EnvironmentStatus(
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        paddle_version=paddle_version,
        paddle_cuda_compiled=paddle_cuda_compiled,
        nvidia_packages=nvidia_packages,
    )


def status_as_dict(status: EnvironmentStatus) -> dict[str, Any]:
    """Convert an environment status object to a plain dictionary."""

    return {
        "torch_version": status.torch_version,
        "torch_cuda_available": status.torch_cuda_available,
        "paddle_version": status.paddle_version,
        "paddle_cuda_compiled": status.paddle_cuda_compiled,
        "nvidia_packages": status.nvidia_packages,
    }
