TODO List
==========

> next checkpoint's agenda:

> remember to work on dev branch
    until you have finished to work with vaccaro's repo,
    then you should create a branch for each feature that you
    would like to implement. Maybe it's worth a check online
    for tools like Jira, but free and easier to use

## high priority
- implement a way to log config
- implement a way to log metrics

- ask by email Marco about ws access
- set up ws
- download BVI-DVC on ws
- train UNet with BVI-DVC
- train SR-UNet with BVI-DVC
- train with/without ssim

- understand why your way of including ssim was so bad for the training
---

## medium priority
- replace tensorboard with mlflow
- check structural reparametrization in DiracNets and RepVGG
- check at line 88 in `vaccaro/pytorch_unet.py` dimensions of `self.conv_adapter.weight`
- the Professor suggests Comet.ml, but I'll check mlflow also since I'm using it at work

---

## low priority
- refactor CustomPyTorchDataset class:
    - rename it ?
    - use function composition
    - extract a transform function removing all internal preprocessing
- pay attention in piq: LPIPS seems to yield different results than in the
    original implementation

---
An alternative to the official BVI-DVC dataset can be found at [https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip](https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip)
Also, note that the original BVI-DVC.zip is 85.7GB, and it contains 800 sequences, and this alternative contains only 772 sequences, and it is 83.8GB big.

## Videos from [Derf's collection](https://media.xiph.org/video/derf/) used
by Vaccaro in his paper as an extra test dataset:
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

#### DONEs
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
- DONE - study SR-UNet
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

