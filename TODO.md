TODO List
==========

> remember to work on dev branch
> remember to activate ~/.venv/binarization/bin/activate
> remember to test with notebooks only in ~/Projects/test-binarization

## high priority

- choose an implementation for the unet
- understand better the changes that has been done on the unet from Vaccaro
- understand the details of the GAN framework in Vaccaro's repo and the
    differences with the No-GAN approach in Mameli's repo.
- train and validate something

- study and translate [...]/fast-sr-unet/models.py (UNet)

- study and translate [...]/fast-sr-unet/train.py
- study and translate [...]/fast-sr-unet/models.py (ConvolutionalBlock)
- study and translate [...]/fast-sr-unet/models.py (Discriminator)

---

## medium priority
- refactor CustomPyTorchDataset class
    - rename it ?
    - use function composition
- Hydra set-up

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

