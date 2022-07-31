TODO List
==========

> continue on ./binarization/future\_training.py
> remember to work on dev branch

> next checkpoint:
    - ??

## high priority
- A/B test with previous ssim loss
- check structural reparametrization in DiracNets and RepVGG
- check at line 88 in vaccaro/pytorch\_unet.py dimensions of self.conv\_adapter.weight
- implement SR-UNet
- as soon as you got the Wi-Fi, download the dataset from [here](https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip)
    - (the original BVI-DVC.zip is 85.7GB, and it contains 800 sequences)
    - (this other one instead contains 772 sequences, and it is 83.8GB big)
- try to run training with sr-unet (by vaccaro)

- understand the details of the GAN framework in Vaccaro's repo and the
    differences with the No-GAN approach in Mameli's repo.

---

## medium priority
- replace scaling 540 -> 512 (270 -> 256), with black bands
- set up [Hydra Structured Config](https://hydra.cc/docs/advanced/terminology/#structured-config)
- use ./binarization/models/ to store each architecture in a single file
- refactor CustomPyTorchDataset class:
    - rename it ?
    - use function composition
    - extract a transform function removing all internal preprocessing
- the Professor suggests Comet.ml, but I'll check mlflow since I'm using it at work

---

## low priority
- pay attention in piq: LPIPS seems to yield different results than in the
    original implementation

---

## Videos from [Derf's collection](https://media.xiph.org/video/derf/) used
by Vaccaro in his paper as an extra test dataset:
+ -> downloaded on Alienware-M15
- -> not yet downloaded on Alienware-M15
    + ducks_take_off
    + crowd_run
    + controlled_burn
    + aspen
    + snow_mnt
    + touchdown_pass
    + station2
    + rush_hour
    + blue_sky
    + riverbed
    + old_town_cross
    + rush_field_cuts
    + in_to_tree
    + sunflower

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
- DONE - understand difference between vaccoro/pytorch\_ssim and piq.SSIMLoss,
    especially why the latter doesn't handle input values < 0, while the
    former can.
- DONE - do some experimentation with the UNet
- DONE - study SR-UNet

