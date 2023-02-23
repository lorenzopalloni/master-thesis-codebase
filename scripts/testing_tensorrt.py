"""Script to evaluate an image with a super-resolution model"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from binarization.config import Gifnoc, get_default_config
from binarization.dataset import CalibrationDataset, get_test_batches
from binarization.datatools import (
    draw_validation_fig,
    make_4times_downscalable,
    postprocess,
)
from binarization.traintools import prepare_cuda_device, prepare_generator

if __name__ == "__main__":
    user_cfg = get_default_config()
    user_cfg.params.buffer_size = 1

    unet_ckpt_path = Path(
        user_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_19_srunet_4_318780.pth",
    )

    user_cfg.model.ckpt_path_to_resume = unet_ckpt_path
    user_cfg.model.name = 'srunet'

    device_id = 0
    n_evaluations = 1
    cfg = user_cfg

    ckpt_path = cfg.model.ckpt_path_to_resume
    save_dir = cfg.paths.outputs_dir / ckpt_path.stem
    save_dir.mkdir(exist_ok=True)

    device = prepare_cuda_device(device_id=device_id)
    gen = prepare_generator(user_cfg, device=device)
    # from binarization.models import UNet
    # gen = UNet()

    gen.to(device)

    from torch.utils.data import DataLoader

    calibration_dataset = CalibrationDataset(cfg)

    from torch.utils.data import Subset

    sub_size = 5
    sub_calibration_dataset = Subset(calibration_dataset, range(sub_size))

    calibration_dataloader = DataLoader(
        dataset=sub_calibration_dataset,
        batch_size=1,
        shuffle=False,
    )

    import torch_tensorrt

    gen.eval()
    inputs = [torch_tensorrt.Input((1, 3, 288, 480), dtype=torch.float)]
    enabled_precisions = {torch.int8}  # Run with fp16

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        calibration_dataloader,
        cache_file="./calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )
    tensorrt_device = {
        "device_type": torch_tensorrt.DeviceType.GPU,
        "gpu_id": 0,
        "dla_core": 0,
        "allow_gpu_fallback": False,
        "disable_tf32": False,
    }

    trt_ts_module = torch_tensorrt.compile(
        module=gen,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        calibrator=calibrator,
        device=tensorrt_device,
    )
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

    # progress_bar = tqdm(calibration_dataloader, total=n_evaluations)

    # for step_id, (original, compressed) in enumerate(progress_bar):
    #     if n_evaluations and step_id > n_evaluations - 1:
    #         break

    #     compressed = compressed.to(device)
    #     compressed = make_4times_downscalable(compressed)

    #     import torch_tensorrt
    #     gen.eval()
    #     inputs = [torch_tensorrt.Input((1, 3, 288, 480), dtype=torch.float)]
    #     enabled_precisions = {torch.float, torch.int8}  # Run with fp16

    #     calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    #         test_batches,
    #         cache_file="./calibration.cache",
    #         use_cache=False,
    #         algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    #         device=torch.device("cuda:0"),
    #     )
    #     tensorrt_device={
    #         "device_type": torch_tensorrt.DeviceType.GPU,
    #         "gpu_id": 0,
    #         "dla_core": 0,
    #         "allow_gpu_fallback": False,
    #         "disable_tf32": False
    #     }

    #     trt_ts_module = torch_tensorrt.compile(
    #         module=gen,
    #         inputs=inputs,
    #         enabled_precisions=enabled_precisions,
    #         calibrator=calibrator,
    #         device=tensorrt_device,
    #     )
    #     torch.jit.save(trt_ts_module, "trt_ts_module.ts")

    #     # input_data = compressed
    #     # result = trt_ts_module(input_data)

    #     with torch.no_grad():
    #         generated = gen(compressed)

    # compressed = compressed.cpu()
    # generated = generated.cpu()
    # generated = postprocess(original=original, generated=generated)

    # for i in range(original.shape[0]):
    #     fig = draw_validation_fig(
    #         original_image=original[i],
    #         compressed_image=compressed[i],
    #         generated_image=generated[i],
    #     )
    #     fig.savefig(save_dir / f'{step_id:05d}_validation_fig.jpg')
    #     plt.close(fig)  # close the current fig to prevent OOM issues

# model = MyModel().eval()  # torch module needs to be in eval (not training) mode

# inputs = [
#     torch_tensorrt.Input(
#         min_shape=[1, 1, 16, 16],
#         opt_shape=[1, 1, 32, 32],
#         max_shape=[1, 1, 64, 64],
#         dtype=torch.half,
#     )
# ]
# enabled_precisions = {torch.float, torch.half}  # Run with fp16

# trt_ts_module = torch_tensorrt.compile(
#     model, inputs=inputs, enabled_precisions=enabled_precisions
# )

# input_data = input_data.to("cuda").half()
# result = trt_ts_module(input_data)
# torch.jit.save(trt_ts_module, "trt_ts_module.ts")

# # Deployment application
# import torch
# import torch_tensorrt

# trt_ts_module = torch.jit.load("trt_ts_module.ts")
# input_data = input_data.to("cuda").half()
# result = trt_ts_module(input_data)

##############################################################################
# testing_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
#     ),
# )

# testing_dataloader = torch.utils.data.DataLoader(
#     testing_dataset, batch_size=1, shuffle=False, num_workers=1
# )
# calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
#     testing_dataloader,
#     cache_file="./calibration.cache",
#     use_cache=False,
#     algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
#     device=torch.device("cuda:0"),
# )

# trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 32, 32))],
#                                     enabled_precisions={torch.float, torch.half, torch.int8},
#                                     calibrator=calibrator,
#                                     device={
#                                          "device_type": torch_tensorrt.DeviceType.GPU,
#                                          "gpu_id": 0,
#                                          "dla_core": 0,
#                                          "allow_gpu_fallback": False,
#                                          "disable_tf32": False
#                                      })
##############################################################################
