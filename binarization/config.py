# pylint: disable=missing-module-docstring,missing-class-docstring,too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gifnoc import Gifnoc


@dataclass
class ModelConfig:
    name: str | None = None  # {'unet', 'srunet'}
    num_filters: int = 64
    use_residual: bool = True
    use_batch_norm: bool = False
    scale_factor: int = 4
    ckpt_path_to_resume: Path | None = None


@dataclass
class ParamsConfig:
    dis_lr: float = 1e-4
    gen_lr: float = 1e-4
    patch_size: int = 96
    batch_size: int = 32  # {8, 32}
    buffer_size: int = 32
    n_batches_per_buffer: int = 128
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    save_ckpt_every: int = 20_000
    num_workers: int = 1
    num_epochs: int = 3
    w0: float = 1e-0  # LPIPS weight
    w1: float = 1e-0  # SSIM weight
    w2: float = 1e-3  # Adversarial loss weight


@dataclass
class PathsConfig:
    project_dir: Path = Path(__file__).parent.parent
    outputs_dir: Path = project_dir / "outputs"
    artifacts_dir: Path = project_dir / "artifacts"
    mlruns_dir: Path = artifacts_dir / 'mlruns'
    data_dir: Path = project_dir / "data"
    original_frames_dir: Path = data_dir / "original_frames"
    compressed_frames_dir: Path = data_dir / "compressed_frames"


@dataclass
class DefaultConfig:
    params: ParamsConfig = ParamsConfig()
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()


def get_default_config() -> Gifnoc:
    """Instantiates a config with default values"""
    return Gifnoc.from_dataclass(DefaultConfig())


if __name__ == "__main__":
    paths = PathsConfig()
    print(paths)
    print(Gifnoc.from_dataclass(paths))
