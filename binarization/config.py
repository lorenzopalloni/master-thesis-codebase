# pylint: disable=missing-module-docstring,missing-class-docstring,too-many-instance-attributes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gifnoc import Gifnoc


@dataclass
class UNetConfig:
    num_filters: int = 64
    use_residual: bool = True
    use_batch_norm: bool = False
    scale_factor: int = 2
    ckpt_path_to_resume: Optional[Path] = None
    starting_epoch_id: int = 0


@dataclass
class SRUNetConfig:
    num_filters: int = 64
    use_residual: bool = True
    use_batch_norm: bool = False
    scale_factor: int = 2
    ckpt_path_to_resume: Optional[Path] = None


@dataclass
class ParamsConfig:
    dis_lr: float = 1e-4
    gen_lr: float = 1e-4
    patch_size: int = 96
    batch_size: int = 16  # {8, 32}
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    num_workers: int = 1  # {1, 12}
    num_epochs: int = 100
    w0: float = 1e-0  # LPIPS weight
    w1: float = 1e-0  # SSIM weight
    w2: float = 1e-3  # Adversarial loss weight
    active_model_name: str = 'unet'
    unet: UNetConfig = UNetConfig()
    srunet: SRUNetConfig = SRUNetConfig()


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


def get_default_config() -> Gifnoc:
    """Instantiates a config with default values"""
    return Gifnoc.from_dataclass(DefaultConfig())


if __name__ == "__main__":
    paths = PathsConfig()
    print(paths)
    print(Gifnoc.from_dataclass(paths))
