from pathlib import Path
from datetime import datetime

import torch
import piq
import lpips
import tqdm
import fire
from torch.utils.tensorboard import SummaryWriter

from binarization import models
from binarization import dataset


def main(project_dir: Path):
    # general hyperparameters
    dis_lr = 1e-4
    gen_lr = 1e-4
    patch_size = 96
    batch_size = 8
    num_workers = 8  # {1, 12}
    num_epochs = 20
    w0 = 1e-0  # LPIPS weight
    w1 = 1e-0  # SSIM weight
    w2 = 1e-3  # Adversarial loss weight

    # unet hyperparameters
    num_filters = 64
    use_residual = True
    use_batch_norm = False
    scale_factor: int = 4

    # data configuration
    # project_dir = Path(__file__).parent.parent
    project_dir = Path(project_dir)
    data_dir = project_dir / 'data'
    train_dir = data_dir / 'train'
    # val_dir = data_dir / 'val'
    # test_dir = data_dir / 'test'
    train_original_frames_dir = train_dir / 'original_frames'
    train_encoded_frames_dir = train_dir / 'encoded_frames'
    checkpoints_dir = project_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

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
        original_frames_dir=train_original_frames_dir,
        encoded_frames_dir=train_encoded_frames_dir,
        patch_size=patch_size,
        training=True,
    )
    # ds_val = dataset.CustomPyTorchDataset(
    #     original_frames_dir=val_original_frames_dir,
    #     encoded_frames_dir=val_encoded_frames_dir,
    #     patch_size=patch_size,
    #     training=False,
    # )

    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    # dl_val = torch.utils.data.DataLoader(
    #     dataset=ds_val,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     shuffle=True,
    #     pin_memory=True,
    # )

    for epoch_id in range(num_epochs):
        progress_bar = tqdm.tqdm(dl_train)
        for step_id, (lq, hq) in enumerate(progress_bar):
            gen.train()
            dis.train()

            # discriminator training step
            ##################################################################
            dis_optim.zero_grad()

            lq = lq.to(device)
            hq = hq.to(device)
            generated_hq = gen(lq)

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

            # generator training step
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(generated_hq, hq).mean()

            x_min = min(generated_hq.min(), hq.min())
            x_max = max(generated_hq.max(), hq.max())
            loss_ssim = 1.0 - ssim_loss_op(
                dataset.min_max_scaler(generated_hq, x_min, x_max),
                dataset.min_max_scaler(hq, x_min, x_max)
            )
            
            pred_generated_hq = dis(generated_hq)
            loss_bce = bce_loss_op(
                pred_generated_hq,
                torch.ones_like(pred_generated_hq)
            )
            loss_gen = w0 * loss_lpips + w1 * loss_ssim + w2 * loss_bce

            loss_gen.backward()
            gen_optim.step()
            ##################################################################

            # logging statistics on training set
            ##################################################################
            progress_bar.set_description(
                'Epoch #{} - '
                'Loss dis: {:.8f}; '
                'Loss gen: {:.4f} = '
                '({:.4f} + {:.4f} * {:.4f})'.format(
                    epoch_id,
                    float(loss_dis),
                    float(loss_gen),
                    float(w0 * loss_lpips + w1 * loss_ssim),
                    float(w2),
                    float(loss_bce),
                )
            )
            ##################################################################
        
        str_now = datetime.now().strftime(r"%Y%m%d%H%M%S")
        torch.save(gen.state_dict(), checkpoints_dir / f"{str_now}.pth")


if __name__ == "__main__":
    fire.Fire(main)
