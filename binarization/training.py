from pathlib import Path

import torch
import piq
import lpips
import tqdm

import models
import dataset

# TODO: Hydra | classes
dis_lr = 1e-4
gen_lr = 1e-4
patch_size = 96
batch_size = 32
num_workers = 1  # {12}
num_epochs = 2
w0, w1, w2 = 1e-0, 1e-0, 1e-3

project_root_dir = Path(__file__).parent.parent
data_dir = project_root_dir / 'data'
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'
original_frames_dir = train_dir / 'original_frames_dir'
encoded_frames_dir = train_dir / 'encoded_frames_dir'

# TODO: implement
gen = models.UNet()
# TODO: implement
dis = models.Discriminator()

dis_optim = torch.optim.Adam(lr=dis_lr, params=dis.parameters())
gen_optim = torch.optim.Adam(lr=gen_lr, params=gen.parameters())

ssim_loss_op = piq.ssim_loss()
lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
lpips_alex_loss_op = lpips.LPIPS(net='alex', version='0.1')
bce_loss_op = torch.nn.BCELoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_vgg_loss_op.to(device)
lpips_alex_loss_op.to(device)
gen.to(device)
dis.to(device)

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
    for step, (x, y_true) in enumerate(tqdm_):
        gen.train()
        dis.train()

        dis_optim.zero_grad()

        x = x.to(device)
        y_true = y_true.to(device)

        y_fake = gen(x)

        # train discriminator phase
        batch_dim = x.shape[0]
        y_pred_true = dis(y_true)

        # TODO: handle name convention values vs functions
        # e.g.: loss_dis / dis_loss, (value / function)

        # forward pass on true
        loss_true = bce_loss_op(y_pred_true, torch.ones_like(y_pred_true))

        # then updates on fakes
        y_pred_fake = dis(y_fake.detach())
        loss_fake = bce_loss_op(y_pred_fake, torch.zeros_like(y_pred_fake))

        loss_dis = loss_true + loss_fake
        loss_dis *= 0.5

        loss_dis.backward()
        dis_optim.step()

        # train generator phase
        gen_optim.zero_grad()

        loss_lpips = lpips_vgg_loss_op(y_fake, y_true).mean()
        loss_ssim = 1.0 - ssim_loss_op(y_fake, y_true)
        y_pred_fake = dis(y_fake)
        loss_bce = bce_loss_op(y_pred_fake, torch.ones_like(y_pred_fake))
        loss_gen = w0 * loss_lpips + w1 * loss_ssim + w2 * loss_bce

        loss_gen.backward()
        gen_optim.step()

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
