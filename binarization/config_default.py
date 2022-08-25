# pylint: disable=missing-module-docstring,missing-class-docstring,too-many-instance-attributes
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class UNetConfig:
    num_filters: int = 64
    use_residual: bool = True
    use_batch_norm: bool = False
    scale_factor: int = 4


@dataclass
class ParamsConfig:
    dis_lr: float = 1e-4
    gen_lr: float = 1e-4
    patch_size: int = 96
    batch_size: int = 8  # {8, 32}
    limit_train_batches: Optional[int] = 2
    limit_val_batches: Optional[int] = 2
    num_workers: int = 4  # {1, 12}
    num_epochs: int = 10
    w0: float = 1e-0  # LPIPS weight
    w1: float = 1e-0  # SSIM weight
    w2: float = 1e-3  # Adversarial loss weight
    unet: UNetConfig = UNetConfig()


@dataclass
class PathsConfig:
    project_dir: Path = Path().resolve()
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
    artifacts_dir: Path = project_dir / "artifacts"


@dataclass
class MainConfig:
    params: ParamsConfig = ParamsConfig()
    paths: PathsConfig = PathsConfig()


if __name__ == "__main__":
    from gifnoc import Gifnoc

    paths = PathsConfig()
    print(paths)
    print(Gifnoc.from_dataclass(paths))
