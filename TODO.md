TODO List
==========

> remember to work on dev branch
> remember to activate ~/.venv/binarization/bin/activate
> remember to test with notebooks only in ~/Projects/test-binarization

## high priority

- try to run training with unet (by vaccaro)
- try to run training with sr-unet (by vaccaro)
- implement your own unet/sr-unet

- use ./binarization/models/ to store each architecture in a single file
- understand the details of the GAN framework in Vaccaro's repo and the
    differences with the No-GAN approach in Mameli's repo.

- study and translate [...]/fast-sr-unet/models.py (ConvolutionalBlock)
- study and translate [...]/fast-sr-unet/models.py (Discriminator)
- study and translate [...]/fast-sr-unet/models.py (UNet)
- study and translate [...]/fast-sr-unet/train.py

---

## medium priority
- set up [Hydra Structured Config](https://hydra.cc/docs/advanced/terminology/#structured-config)
- refactor CustomPyTorchDataset class:
    - rename it ?
    - use function composition
- Professor suggests Comet.ml

---

## low priority
- pay attention in piq: LPIPS seems to yield different results than in the
    original implementation

---

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

