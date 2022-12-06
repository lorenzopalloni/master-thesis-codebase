# TODO List

> resume from `binarization/models/UNet`
> current branch: `train_srunet`

> have a look at **normalization**

> [standby] resume with: `ssh solaris && screen -r binarization`

## high priority

- refactor UNet
- refactor `binarization/models/common.py`
- refactor SRUNet

- train SRUNet

- evaluate with `eval_image.py` both models
- evaluate with `eval_video.py` both models
- implement binarized UNet
- train binarized UNet
- implement binarized SRUNet
- train binarized SRUNet
- evaluate them all

- train with/without ssim

- understand why your way of including ssim was so bad for the training
---

## medium priority
- check structural reparametrization in DiracNets and RepVGG
- check at line 88 in `vaccaro/pytorch_unet.py` dimensions of `self.conv_adapter.weight`

---

## low priority
- take a look in piq: LPIPS seems to yield different results than in the
    original implementation

---
### How to compile torchvision with ffmpeg support:
    1. git clone git@github.com:pytorch/vision.git
    2. sudo apt install ffmpeg
    3. sudo apt install libavcodec-dev
    4. sudo apt install libavfilter-dev
    5. sudo apt install libswscale-dev
    6. pip install --upgrade av==8.1.0
    7. cd ./vision && python3.8 setup.py install

### BVI-DVC alternative
An alternative to the official BVI-DVC dataset can be found at [https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip](https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip)
Also, note that the original BVI-DVC.zip is 85.7GB, and it contains 800 sequences, and this alternative contains only 772 sequences, and it is 83.8GB big.

### Videos from [Derf's collection](https://media.xiph.org/video/derf/) used by Vaccaro in his paper as an extra test dataset:
+ -> downloaded on Alienware-M15
- -> not yet downloaded on Alienware-M15
    + `ducks_take_off`
    + `crowd_run`
    + `controlled_burn`
    + `aspen`
    + `snow_mnt`
    + `touchdown_pass`
    + `station2`
    + `rush_hour`
    + `blue_sky`
    + `riverbed`
    + `old_town_cross`
    + `rush_field_cuts`
    + `in_to_tree`
    + `sunflower`

## DONEs
- DONE - .gitignore literature/ folder, plus Python and LaTex stuff
- DONE - convert sh script for video preparation to Python
- DONE - implement a custom pytorch dataset
- DONE - git set-up
- DONE - create a new dev branch and start working there
- DONE - implement function composition
- DONE - replace fire with argparse
- DONE - fix tests/assets issue
- DONE - init Hydra configuration
- DONE - try to run training with unet (by vaccaro)
- DONE - data are scaled x4, you need to adjust the UNet implementation
- DONE - understand difference between `vaccoro/pytorch_ssim` and `piq.SSIMLoss`,
    especially why the latter doesn't handle input values < 0, while the
    former can.
- DONE - do some experimentation with the UNet
- DONE - study SRUNet
- DONE - set up [Hydra Structured Config](https://hydra.cc/docs/advanced/terminology/#structured-config)
- DONE - log more info while training
- DONE - solve GPU issue in Ubuntu 20.04 (Alienware)
- DONE - replace the scaling 540 -> 512 (or 270 -> 256), with black bands
- DONE - copy-paste SSIM usage from `binarization/vaccaro/train.py` into
    `binarization/future_training.py`, you need to replace `piq.SSIMLoss`,
    with the same function in vaccaro's `train.py` module.
- DONE - as soon as you got the Wi-Fi, download the dataset from [here](https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip)
    - (the original BVI-DVC.zip is 85.7GB, and it contains 800 sequences)
    - (this other one instead contains 772 sequences, and it is 83.8GB big)
- DONE - complete eval.py script, need a way to show progress/results
- DONE - add more `piq.[metric]` on validation step in `future_training.py`
- DONE - rename `future_training.py` -> `train.py`
- DONE - add string-ify method to gifnoc (`cfg.params.unet.num_filters` -> `cfg_params_unet_num_filters`)
- DONE - log params with mlflow
- DONE - replace tensorboard with mlflow
- DONE - set up a couple of sample experiments with mlflow
- DONE - set up ws
- DONE - download BVI-DVC on ws (it was already there)
- DONE - try to work with symlinks
- DONE - set up data
- DONE - learn how to use `screen` command
- DONE - fix: image too big to be loaded and cropped one-by-one
- DONE - add in config `buffer_size` and `n_batches_per_buffer`
- DONE - refactor `batch_generator`, from function to class
- DONE - debug data loader, there's something weird
- DONE - symbolic link `ln -s ./data/original_videos` -> `/homes/datasets/BVI_DVC/3h<...>/Videos/`
- DONE - run `./scripts/video_preprocessing.py -i data -s 4`
- DONE - train UNet with `scale_factor == 4`
- DONE - implement `eval_video.py` (video + timing) while UNet is training
