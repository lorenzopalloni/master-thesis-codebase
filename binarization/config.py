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
    batch_size: int = 8  # {8, 32}
    limit_train_batches: Optional[int] = 2
    limit_val_batches: Optional[int] = 2
    num_workers: int = 4  # {1, 12}
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
    train_dir: Path = data_dir / "train"
    val_dir: Path = data_dir / "val"
    test_dir: Path = data_dir / "test"
    train_original_frames_dir: Path = train_dir / "original_frames"
    train_encoded_frames_dir: Path = train_dir / "encoded_frames"
    val_original_frames_dir: Path = val_dir / "original_frames"
    val_encoded_frames_dir: Path = val_dir / "encoded_frames"
    test_original_frames_dir: Path = test_dir / "original_frames"
    test_encoded_frames_dir: Path = test_dir / "encoded_frames"


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
