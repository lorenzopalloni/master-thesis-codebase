TODO List
==========

> resume from the last notebook
> resume from ./binarization/future\_training.py
> remember to work on dev branch

> next checkpoint:
    - ??

## high priority

- I've recently changed dataset.CustomPyTorchDataset.index\_handler behaviour:
    - you should update its related tests


- log more info while training
- A/B test with previous ssim loss

- check structural reparametrization in DiracNets and RepVGG


---

## medium priority
- resume from logging validation images original vs. generated
- check at line 88 in vaccaro/pytorch\_unet.py dimensions of self.conv\_adapter.weight

- as soon as you got the Wi-Fi, download the dataset from [here](https://data.bris.ac.uk/datasets/tar/3h0hduxrq4awq2ffvhabjzbzi1.zip)
    - (the original BVI-DVC.zip is 85.7GB, and it contains 800 sequences)
    - (this other one instead contains 772 sequences, and it is 83.8GB big)

- replace the scaling 540 -> 512 (or 270 -> 256), with black bands
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
- DONE - set up [Hydra Structured Config](https://hydra.cc/docs/advanced/terminology/#structured-config)
