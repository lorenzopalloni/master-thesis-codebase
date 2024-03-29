# TODO List

## high priority

- rename the codebase (suggestions: ??)
- handle the gifnoc package (rename, refactor, then publish on PyPI)

## medium priority
- train again changing the perceptual loss

## low priority

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
- DONE - refactor UNet
- DONE - refactor `binarization/models/common.py`
- DONE - refactor SRUNet
- DONE - be able to train SRUNet
- DONE - fix normalization ([0, 1] -> [-1, 1])
- DONE - fix clamp's
- DONE - fix RGB -> BGR removing conversions
- DONE - fix any bug caused by the previous three points
- DONE - be able to evaluate with `eval_image.py` both models
- DONE - be able to evaluate with `eval_video.py` both models
- DONE - check at line 88 in `vaccaro/pytorch_unet.py` dimensions of `self.conv_adapter.weight`
- DONE - implement an eval script taking a cue from `fede-vaccaro/fast-sr-unet/evaluation_model.py`
- DONE - script to compile unet/srunet in int8/fp16/fp32 TensorRT
- DONE - keep track of average inference times in `scripts/eval_image.py`
- DONE - add fp16/fp32 to results visualization in `notebooks/2022_02_28_model_evaluation.ipynb`
- DONE - draw plots about average inference times
- DONE - write an email to the prof including Leo to show results
- DONE - start writing the thesis
- DONE - implement a pipeline to:
    - compress/scale an original video
    - `video_to_frames` on compressed video
    - `video_to_frames` on original video
    - `eval_images` generate frames from compressed frames
    - `frames_to_video` on original frames
    - `frames_to_video` on generated frames
    - `vmaf` between original, and generated videos
- DONE - pipeline with inputs (original video, model) -> VMAF score
- DONE - eval models with VMAF
- DONE - train again with original frames in .png instead of .jpg

