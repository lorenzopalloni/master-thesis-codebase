"""Script to evaluate a trained super-resolution model"""

from pathlib import Path

import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from binarization import dataset, train
from binarization.config import Gifnoc, default_config


def inv_adjust_image_for_unet(
    generated: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    hy, wy = generated.shape[-2], generated.shape[-1]
    hx, wx = original.shape[-2], original.shape[-1]
    return F.crop(generated, (hy - hx) // 2, (wy - wx) // 2, hx, wx)


def process_raw_generated(
    generated: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    """Postprocesses outputs from super-resolution generator models"""
    out = generated
    out = dataset.inv_min_max_scaler(out)
    out = out.clip(0, 255)
    out = out / 255.0
    out = inv_adjust_image_for_unet(out, original)
    return out


def main(cfg: Gifnoc):
    save_dir = cfg.paths.outputs_dir / cfg.params.ckpt_path_to_resume.stem
    save_dir.mkdir(exist_ok=True, parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whole_images_dataset = dataset.WholeImagesDataset(
        original_frames_dir=cfg.paths.val_original_frames_dir,
        encoded_frames_dir=cfg.paths.val_encoded_frames_dir,
    )
    dl_val = dataset.DataLoader(
        dataset=whole_images_dataset,
        batch_size=cfg.params.batch_size,
        shuffle=None,
    )
    progress_bar_val = tqdm(dl_val)
    gen = train.set_up_unet(cfg)
    gen.to(device)
    counter = 0
    for step_id_val, (compressed_val, original_val) in enumerate(
        progress_bar_val
    ):

        compressed_val = compressed_val.to(device)
        original_val = original_val.to(device)

        compressed_val = dataset.adjust_image_for_unet(compressed_val)

        gen.eval()
        with torch.no_grad():
            generated_val = gen(compressed_val)

        original_val = original_val.cpu()
        generated_val = generated_val.cpu()
        compressed_val = compressed_val.cpu()
        generated_val = process_raw_generated(generated_val, original_val)

        for i in range(original_val.shape[0]):
            fig = dataset.draw_validation_fig(
                original_image=original_val[i],
                compressed_image=compressed_val[i],
                generated_image=generated_val[i],
            )
            save_path = save_dir / f'validation_fig_{counter}.jpg'
            counter += 1
            fig.savefig(save_path)
            # TODO: close fig to save memory


if __name__ == "__main__":
    # default_config.params.ckpt_path_to_resume = Path('/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_28_epoch_9.pth')
    default_config.params.ckpt_path_to_resume = Path(
        '/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_31_epoch_13.pth'
    )
    default_config.params.batch_size = 10
    main(default_config)
