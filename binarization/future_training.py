"""Training module"""

from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import piq
import torch
import torchvision.transforms.functional as F
from gifnoc import Gifnoc
from tqdm import tqdm

from binarization import dataset, models, utils
from binarization.config_default import MainConfig
from binarization.vaccaro import pytorch_ssim


def main(cfg: Gifnoc):

    specific_artifacts_dir: Path = utils.make_specific_artifacts_dir(
        artifacts_dir=Path(cfg.paths.artifacts_dir)
    )
    tensorboard_logger = utils.create_tensorboard_logger(
        specific_artifacts_dir=specific_artifacts_dir
    )
    checkpoints_dir = specific_artifacts_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    gen = models.UNet(
        num_filters=cfg.params.unet.num_filters,
        use_residual=cfg.params.unet.use_residual,
        use_batch_norm=cfg.params.unet.use_batch_norm,
        scale_factor=cfg.params.unet.scale_factor,
    )

    dis = models.Discriminator()

    dis_optim = torch.optim.Adam(lr=cfg.params.dis_lr, params=dis.parameters())
    gen_optim = torch.optim.Adam(lr=cfg.params.gen_lr, params=gen.parameters())

    # ssim_loss_op = piq.SSIMLoss()
    ssim_loss_op = pytorch_ssim.SSIM()
    lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex_loss_op = lpips.LPIPS(net='alex', version='0.1')
    bce_loss_op = torch.nn.BCELoss()

    # Move models and losses to cuda device if any is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen.to(device)
    dis.to(device)
    lpips_vgg_loss_op.to(device)
    lpips_alex_loss_op.to(device)
    print(f'current device: {device}')

    dl_train, dl_val, dl_test = dataset.make_dataloaders(
        cfg=cfg,
        patch_size=cfg.params.patch_size,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
    )

    global_step_id = 0
    for epoch_id in range(cfg.params.num_epochs):
        progress_bar_train = tqdm(
            dl_train, total=cfg.params.limit_train_batches
        )
        for step_id_train, (compressed_patches, original_patches) in enumerate(
            progress_bar_train
        ):
            if (
                cfg.params.limit_train_batches is not None
                and step_id_train > cfg.params.limit_train_batches
            ):
                break
            gen.train()
            # dis.train()

            # Discriminator training step
            ##################################################################
            dis_optim.zero_grad()

            compressed_patches = compressed_patches.to(device)
            original_patches = original_patches.to(device)
            generated_patches = gen(compressed_patches)

            pred_original_patches = dis(original_patches)

            loss_true = bce_loss_op(
                pred_original_patches, torch.ones_like(pred_original_patches)
            )

            pred_generated_patches = dis(generated_patches.detach())
            loss_fake = bce_loss_op(
                pred_generated_patches,
                torch.zeros_like(pred_generated_patches),
            )

            loss_dis = loss_true + loss_fake
            loss_dis *= 0.5

            loss_dis.backward()
            dis_optim.step()
            ##################################################################

            # generator training step
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(
                generated_patches, original_patches
            ).mean()

            # x_min = min(
            #     generated_patches.min(), original_patches.min()
            # )
            # x_max = max(
            #     generated_patches.max(), original_patches.max()
            # )
            # loss_ssim = 1.0 - ssim_loss_op(
            #     dataset.min_max_scaler(
            #         generated_patches, x_min, x_max
            #     ),
            #     dataset.min_max_scaler(original_patches, x_min, x_max),
            # )

            loss_ssim = 1.0 - ssim_loss_op(generated_patches, original_patches)

            pred_generated_patches = dis(generated_patches)
            loss_bce = bce_loss_op(
                pred_generated_patches,
                torch.ones_like(pred_generated_patches),
            )
            loss_gen = (
                cfg.params.w0 * loss_lpips
                + cfg.params.w1 * loss_ssim
                + cfg.params.w2 * loss_bce
            )

            loss_gen.backward()
            gen_optim.step()
            ##################################################################

            # logging statistics on training set
            ##################################################################
            progress_bar_train.set_description(
                f'Epoch #{epoch_id} - '
                f'Loss dis: {float(loss_dis):.8f}; '
                f'Loss gen: {float(loss_gen):.4f} = '
                f'w0 * {float(loss_lpips):.4f}'
                f' + w1 * {float(loss_ssim):.4f}'
                f' + w2 * {float(loss_bce):.4f})'
            )
            ##################################################################
            tensorboard_logger.add_scalar(
                'lossD', scalar_value=loss_dis, global_step=global_step_id
            )
            tensorboard_logger.add_scalar(
                'lossG', scalar_value=loss_gen, global_step=global_step_id
            )
            # tensorboard_logger.add_image('output_example', img_tensor=<insert-image-here>, global_step=epoch_id * step_id)

            global_step_id += 1

        progress_bar_val = tqdm(dl_val, total=cfg.params.limit_val_batches)
        for step_id_val, (
            compressed_patches_val,
            original_patches_val,
        ) in enumerate(progress_bar_val):
            if (
                cfg.params.limit_val_batches is not None
                and step_id_val > cfg.params.limit_val_batches
            ):
                break
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
                    unprocessed_generated_patch_val = gen(
                        preprocessed_compressed_patch_val
                    )
                    generated_patch_val = F.crop(
                        dataset.inv_min_max_scaler(
                            unprocessed_generated_patch_val.cpu()
                        )
                        .clip(0, 255)
                        .squeeze()
                        / 255.0,
                        0,
                        0,
                        original_patch_val.shape[-2],
                        original_patch_val.shape[-1],
                    )

                fig = dataset.draw_validation_fig(
                    original_image=original_patch_val,
                    compressed_image=compressed_patch_val,
                    generated_image=generated_patch_val,
                )

                tensorboard_logger.add_figure(
                    tag=f"epoch-{epoch_id:03d}-step_id_val-{step_id_val:03d}",
                    figure=fig,
                    global_step=global_step_id,
                )
                generated_patches_val_list.append(generated_patch_val)

            generated_patches_val = torch.stack(generated_patches_val_list)
            ssim_val = piq.ssim(original_patches_val, generated_patches_val)
            print(ssim_val)

            tensorboard_logger.add_scalar(
                "ssim_val", scalar_value=ssim_val, global_step=global_step_id
            )

        # torch.save(
        #     gen.state_dict(),
        #     Path(
        #         checkpoints_dir,
        #         "epoch-{}-lossD-{:.2f}-lossG-{:.2f}.pth".format(
        #             epoch_id, loss_dis, loss_gen
        #         ),
        #     ),
        # )


if __name__ == "__main__":
    cfg = Gifnoc.from_dataclass(MainConfig())
    main(cfg)
