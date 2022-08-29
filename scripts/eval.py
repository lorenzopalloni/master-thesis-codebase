"""Script to evaluate a trained super-resolution model"""

import torch
from tqdm import tqdm

from binarization import dataset, future_training
from binarization.config import Gifnoc, default_config


def main(cfg: Gifnoc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_val = dataset.make_val_dataloader(cfg)
    progress_bar_val = tqdm(dl_val)
    gen = future_training.get_unet(cfg)
    for step_id_val, (
        compressed_patches_val,
        original_patches_val,
    ) in enumerate(progress_bar_val):

        compressed_patches_val.to(device)
        original_patches_val.to(device)

        generated_patches_val_list = []
        for batch_id_val in range(original_patches_val.shape[0]):
            compressed_patch_val = compressed_patches_val[batch_id_val]
            original_patch_val = original_patches_val[batch_id_val]
            preprocessed_compressed_patch_val = (
                dataset.adjust_image_for_unet(compressed_patch_val)
                .unsqueeze(0)
                .to(device)
            )

            gen.eval()
            with torch.no_grad():
                raw_generated_patch_val = gen(
                    preprocessed_compressed_patch_val
                )
                generated_patch_val = future_training.process_raw_generated(
                    raw_generated_patch_val, original_patch_val
                )

            generated_patches_val_list.append(generated_patch_val)

        fig = dataset.draw_validation_fig(
            original_image=original_patch_val,
            compressed_image=compressed_patch_val,
            generated_image=generated_patch_val,
        )
        # fig.savefig


if __name__ == "__main__":
    main(default_config)
