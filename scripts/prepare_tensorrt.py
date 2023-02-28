"""Compile UNet and SRUNet with torch_tensorrt"""

from __future__ import annotations

from pathlib import Path

import torch
import torch_tensorrt

from binarization.config import get_default_config
from binarization.dataset import get_calibration_dataloader
from binarization.traintools import prepare_cuda_device, prepare_generator


def main(model_name: str = "unet", calibration_dataset_size: int = 10):
    assert model_name in {"unet", "srunet"}

    default_cfg = get_default_config()

    device = prepare_cuda_device()
    calibration_dataloader = get_calibration_dataloader(
        cfg=default_cfg,
        subset_size=calibration_dataset_size,
    )

    cfg = default_cfg.copy()
    cfg.model.name = model_name
    cfg.model.ckpt_path_to_resume = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        f"2022_12_19_{cfg.model.name}_4_318780.pth",
    )

    model = prepare_generator(cfg, device=device)
    model.eval()

    tensorrt_inputs = [
        torch_tensorrt.Input((1, 3, 288, 480), dtype=torch.float)
    ]
    tensorrt_enabled_precisions = {torch.int8}  # Run with fp16

    tensorrt_calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        calibration_dataloader,
        cache_file=cfg.paths.trt_dir / f"{cfg.model.name}_calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=device,
    )

    tensorrt_device = {
        "device_type": torch_tensorrt.DeviceType.GPU,
        "gpu_id": 0,
        "dla_core": 0,
        "allow_gpu_fallback": False,
        "disable_tf32": False,
    }

    model_trt = torch_tensorrt.compile(
        module=model,
        inputs=tensorrt_inputs,
        enabled_precisions=tensorrt_enabled_precisions,
        calibrator=tensorrt_calibrator,
        device=tensorrt_device,
    )

    torch.jit.save(model_trt, cfg.paths.trt_dir / f"{cfg.model.name}.ts")


if __name__ == "__main__":
    main("srunet")
