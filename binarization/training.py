from pathlib import Path

import torch
import piq
import lpips
import tqdm
import fire

from binarization import models
from binarization import dataset


def main(project_root_dir: Path):
    # general hyperparameters
    dis_lr = 1e-4
    gen_lr = 1e-4
    patch_size = 96
    batch_size = 32
    num_workers = 1  # {1, 12}
    num_epochs = 2
    w0 = 1e-0  # LPIPS weight
    w1 = 1e-0  # SSIM weight
    w2 = 1e-3  # Adversarial loss weight

    # unet hyperparameters
    num_filters = 64
    use_residual = True  # ??
    use_batch_norm = False  # ??
    scale_factor = 2.0  # ??

    # data configuration
    # project_root_dir = Path(__file__).parent.parent
    data_dir = Path(project_root_dir) / 'data'
    train_dir = data_dir / 'experimenting'  # 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    original_frames_dir = train_dir / 'original_frames'
    encoded_frames_dir = train_dir / 'encoded_frames'

    gen = models.UNet(
        num_filters=num_filters,
        use_residual=use_residual,
        use_batch_norm=use_batch_norm,
        scale_factor=scale_factor,
    )

    dis = models.Discriminator()

    dis_optim = torch.optim.Adam(lr=dis_lr, params=dis.parameters())
    gen_optim = torch.optim.Adam(lr=gen_lr, params=gen.parameters())

    ssim_loss_op = piq.SSIMLoss()
    lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex_loss_op = lpips.LPIPS(net='alex', version='0.1')
    bce_loss_op = torch.nn.BCELoss()

    # move models and losses to cuda device if any is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen.to(device)
    dis.to(device)
    lpips_vgg_loss_op.to(device)
    lpips_alex_loss_op.to(device)
    print(f'current device: {device}')

    # define datasets and data loaders
    ds_train = dataset.CustomPyTorchDataset(
        original_frames_dir=original_frames_dir,
        encoded_frames_dir=encoded_frames_dir,
        patch_size=patch_size,
        training=True,
    )
    ds_val = dataset.CustomPyTorchDataset(
        original_frames_dir=original_frames_dir,
        encoded_frames_dir=encoded_frames_dir,
        patch_size=patch_size,
        training=False,
    )

    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    for epoch_id in range(num_epochs):
        tqdm_ = tqdm.tqdm(dl_train)
        for step, (lq, hq) in enumerate(tqdm_):
            gen.train()
            dis.train()

            # train discriminator phase
            ##################################################################
            dis_optim.zero_grad()

            lq = lq.to(device)
            hq = hq.to(device)
            generated_hq = gen(lq).abs()

            pred_hq = dis(hq)

            loss_true = bce_loss_op(pred_hq, torch.ones_like(pred_hq))

            pred_generated_hq = dis(generated_hq.detach())
            loss_fake = bce_loss_op(
                pred_generated_hq,
                torch.zeros_like(pred_generated_hq)
            )

            loss_dis = loss_true + loss_fake
            loss_dis *= 0.5

            loss_dis.backward()
            dis_optim.step()
            ##################################################################

            # train generator phase
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(generated_hq, hq).mean()
            loss_ssim = 1.0 - ssim_loss_op(generated_hq, hq)
            pred_generated_hq = dis(generated_hq)
            loss_bce = bce_loss_op(
                pred_generated_hq,
                torch.ones_like(pred_generated_hq)
            )
            loss_gen = w0 * loss_lpips + w1 * loss_ssim + w2 * loss_bce

            loss_gen.backward()
            gen_optim.step()
            ##################################################################

            # training log
            ##################################################################
            tqdm_.set_description(
                'Loss dis: {:.6f}; '
                'Loss gen: {:.6f}; '
                'Loss gen (w0 * LPIPS + w1 * SSIM): {:.6f}'
                'Loss gen (BCE only): {:.6f}'.format(
                    float(loss_dis),
                    float(loss_gen),
                    float(w0 * loss_lpips + w1 * loss_ssim + w2),
                    float(loss_bce),
                )
            )
            ##################################################################


if __name__ == "__main__":
    fire.Fire(main)
