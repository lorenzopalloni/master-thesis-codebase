from pathlib import Path
from typing import Optional

import hydra
import lpips
import matplotlib.pyplot as plt
import piq
import torch
import torchvision.transforms.functional as F
from config import MainConfig
from omegaconf import DictConfig
from tqdm import tqdm

from binarization import dataset, models, utils

cs = hydra.core.config_store.ConfigStore()
cs.store(name="config_schema", node=MainConfig)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):

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

    # dis_optim = torch.optim.Adam(lr=cfg.params.dis_lr, params=dis.parameters())
    # gen_optim = torch.optim.Adam(lr=cfg.params.gen_lr, params=gen.parameters())

    # ssim_loss_op = piq.SSIMLoss()
    # lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    # lpips_alex_loss_op = lpips.LPIPS(net='alex', version='0.1')
    # bce_loss_op = torch.nn.BCELoss()

    # # Move models and losses to cuda device if any is available
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # gen.to(device)
    # dis.to(device)
    # lpips_vgg_loss_op.to(device)
    # lpips_alex_loss_op.to(device)
    # print(f'current device: {device}')

    dl_train = dataset.make_train_dataloader(cfg)
    dl_val = dataset.make_val_dataloader(cfg)
    # dl_test = dataset.make_test_dataloader(cfg)

    global_step_id = 0
    for epoch_id in range(cfg.params.num_epochs):
        progress_bar_train = tqdm(
            dl_train, total=cfg.params.limit_train_batches
        )
        # TODO: refactor lq -> compressed_patches, hq -> original_patches
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

            # # Discriminator training step
            # ##################################################################
            # dis_optim.zero_grad()

            # lq = lq.to(device)
            # hq = hq.to(device)
            # generated_hq = gen(lq)

            # pred_hq = dis(hq)

            # loss_true = bce_loss_op(pred_hq, torch.ones_like(pred_hq))

            # pred_generated_hq = dis(generated_hq.detach())
            # loss_fake = bce_loss_op(
            #     pred_generated_hq, torch.zeros_like(pred_generated_hq)
            # )

            # loss_dis = loss_true + loss_fake
            # loss_dis *= 0.5

            # loss_dis.backward()
            # dis_optim.step()
            # ##################################################################

            # # generator training step
            # ##################################################################
            # gen_optim.zero_grad()

            # loss_lpips = lpips_vgg_loss_op(generated_hq, hq).mean()

            # x_min = min(generated_hq.min(), hq.min())
            # x_max = max(generated_hq.max(), hq.max())
            # loss_ssim = 1.0 - ssim_loss_op(
            #     dataset.min_max_scaler(generated_hq, x_min, x_max),
            #     dataset.min_max_scaler(hq, x_min, x_max),
            # )

            # pred_generated_hq = dis(generated_hq)
            # loss_bce = bce_loss_op(
            #     pred_generated_hq, torch.ones_like(pred_generated_hq)
            # )
            # loss_gen = (
            #     cfg.params.w0 * loss_lpips
            #     + cfg.params.w1 * loss_ssim
            #     + cfg.params.w2 * loss_bce
            # )

            # loss_gen.backward()
            # gen_optim.step()
            # ##################################################################

            # # logging statistics on training set
            # ##################################################################
            # progress_bar.set_description(
            #     'Epoch #{} - '
            #     'Loss dis: {:.8f}; '
            #     'Loss gen: {:.4f} = '
            #     '({:.4f} + {:.4f} * {:.4f})'.format(
            #         epoch_id,
            #         float(loss_dis),
            #         float(loss_gen),
            #         float(
            #             cfg.params.w0 * loss_lpips + cfg.params.w1 * loss_ssim
            #         ),
            #         float(cfg.params.w2),
            #         float(loss_bce),
            #     )
            # )
            # ##################################################################
            # tensorboard_logger.add_scalar(
            #     'lossD', scalar_value=loss_dis, global_step=global_step_id
            # )
            # tensorboard_logger.add_scalar(
            #     'lossG', scalar_value=loss_gen, global_step=global_step_id
            # )
            # # tensorboard_logger.add_image('output_example', img_tensor=<insert-image-here>, global_step=epoch_id * step_id)
            global_step_id += 1

        progress_bar_val = tqdm(dl_val, total=cfg.params.limit_val_batches)
        for step_id_val, (
            compressed_patches_val,
            original_patches_val,
        ) in enumerate(progress_bar_val):
            for batch_id_val in range(original_patches_val.shape[0]):

                if (
                    cfg.params.limit_val_batches is not None
                    and step_id_val > cfg.params.limit_val_batches
                ):
                    break

                compressed_patch_val = compressed_patches_val[batch_id_val]
                original_patch_val = original_patches_val[batch_id_val]

                _, original_width, original_height = original_patch_val.shape
                print(
                    f'original_height: {original_height}, original_width: {original_width}'
                )
                (
                    _,
                    compressed_width,
                    compressed_height,
                ) = compressed_patch_val.shape
                print(
                    f'encoded_height: {compressed_width}, encoded_width: {compressed_height}'
                )

                # preprocessed_compressed_patch_val = F.resize(
                #     compressed_patch_val,
                #     size=(256, original_width // cfg.params.unet.scale_factor)
                # )

                preprocessed_compressed_patch_val = (
                    dataset.adjust_image_for_unet(compressed_patch_val)
                )

                gen.eval()

                generated_patch_val = gen(
                    preprocessed_compressed_patch_val.unsqueeze(0)
                )

                generated_patch_val_pil = F.to_pil_image(
                    dataset.inv_min_max_scaler(generated_patch_val)
                    .clip(0, 255)
                    .squeeze()
                    / 255.0
                )

                original_patch_val_pil = F.to_pil_image(
                    dataset.inv_min_max_scaler(original_patch_val)
                )
                compressed_patch_val_pil = F.to_pil_image(
                    dataset.inv_min_max_scaler(compressed_patch_val)
                )

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(30, 15)
                )
                ax1.imshow(original_patch_val_pil)
                ax1.set_title('high quality')
                ax2.imshow(compressed_patch_val_pil)
                ax2.set_title('low quality')
                ax3.imshow(original_patch_val_pil)
                ax3.set_title('high quality')
                ax4.imshow(generated_patch_val_pil)
                ax4.set_title('reconstructed')

                tensorboard_logger.add_figure(
                    f"epoch{epoch_id:03d}-step_id_val-{step_id_val:03d}",
                    figure=fig,
                    global_step=global_step_id,
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
    main()
