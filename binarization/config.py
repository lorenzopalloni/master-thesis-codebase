# pylint: disable=missing-module-docstring,missing-class-docstring,too-many-instance-attributes
"""Default configuration settings"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from gifnoc import Gifnoc


@dataclass
class ModelConfig:
    name: str | None = None  # {'unet', 'srunet'}
    num_filters: int = 64
    use_batch_norm: bool = False
    ckpt_path_to_resume: Path | None = None


@dataclass
class ParamsConfig:
    dis_lr: float = 1e-4
    gen_lr: float = 1e-4
    patch_size: int = 96
    batch_size: int = 14  # {8, 32}
    buffer_size: int = 16
    n_batches_per_buffer: int = 21
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    save_ckpt_every: int = 20_000
    num_workers: int = 1
    num_epochs: int = 3
    lpips_weight: float = 1e-0  # LPIPS weight
    ssim_weight: float = 5e-1  # SSIM weight
    adversarial_loss_weight: float = 1e-3  # Adversarial loss weight
    scale_factor: int = 4
    device: int | str = 0


@dataclass
class PathsConfig:
    project_dir: Path = Path(__file__).parent.parent
    outputs_dir: Path = project_dir / "outputs"
    artifacts_dir: Path = project_dir / "artifacts"
    mlruns_dir: Path = artifacts_dir / 'mlruns'
    data_dir: Path = project_dir / "data"
    original_frames_dir: Path = data_dir / "original_frames"
    compressed_frames_dir: Path = data_dir / "compressed_frames"
    trt_dir: Path = artifacts_dir / "trt"


@dataclass
class DefaultConfig:
    params: ParamsConfig = ParamsConfig()
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()


def get_default_config() -> Gifnoc:
    """Instantiates a config with default values"""
    return Gifnoc.from_dataclass(DefaultConfig())


def parse_args(cfg: Gifnoc) -> Namespace:
    """Parses user-provided arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default=cfg.model.name,
        choices=['unet', 'srunet'],
    )
    parser.add_argument('--device', type=int, default=cfg.params.device)
    parser.add_argument('--ckpt_path_to_resume', type=Path, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    paths = PathsConfig()
    print(paths)
    print(Gifnoc.from_dataclass(paths))

    default_cfg = get_default_config()
    default_cfg.model.name = 'unet'

    example_args = parse_args(cfg=default_cfg)
    print(example_args)
