from dataclasses import dataclass

@dataclass
class UNetConfig:
    num_filters: int
    use_residual: bool
    use_batch_norm: bool
    scale_factor: int

@dataclass
class Params:
    dis_lr: float
    gen_lr: float
    patch_size: int
    batch_size: int
    num_workers: int
    num_epochs: int
    w0: float
    w1: float
    w2: float
    unet: UNetConfig

@dataclass
class Paths:
    project_dir: str
    data_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    train_original_frames_dir: str
    train_encoded_frames_dir: str
    val_original_frames_dir: str
    val_encoded_frames_dir: str
    test_original_frames_dir: str
    test_encoded_frames_dir: str
    artifacts_dir: str

@dataclass
class MainConfig:
    params: Params
    paths: Paths
