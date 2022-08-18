from pathlib import Path
from typing import Optional


class UNetConfig:
    num_filters = 64
    use_residual = True
    use_batch_norm = False
    scale_factor = 4


class Params:
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


class Paths:
    project_dir: Path = Path().resolve()
    data_dir = f"{project_dir}/data"
    # train_dir = "${paths.data_dir}/train"
    # val_dir = "${paths.data_dir}/val"
    # test_dir = "${paths.data_dir}/test"
    # train_original_frames_dir = "${paths.train_dir}/original_frames"
    # train_encoded_frames_dir = "${paths.train_dir}/encoded_frames"
    # val_original_frames_dir = "${paths.val_dir}/original_frames"
    # val_encoded_frames_dir = "${paths.val_dir}/encoded_frames"
    # test_original_frames_dir = "${paths.test_dir}/original_frames"
    # test_encoded_frames_dir = "${paths.test_dir}/encoded_frames"

    # artifacts_dir = "${paths.project_dir}/artifacts"


class MainConfig(dict):
    params: Params = Params()
    paths: Paths = Paths()


if __name__ == '__main__':
    cfg = MainConfig()
    print(cfg)
