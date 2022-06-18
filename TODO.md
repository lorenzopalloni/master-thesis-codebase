TODO List
==========

> remember to activate ~/.venv/binarization/bin/activate
> remember to test with notebooks only in ~/Projects/test-binarization

## high priority

- git set-up
- fix tests/assets issue
- create a new dev branch and start working there
- implement function composition
- refactor CustomPyTorchDataset class
    - rename it ?
    - use function composition
- study and translate [...]/fast-sr-unet/train.py
- Hydra set-up

- files in fast-sr-unet project:
    - data\_loader.py
    - evaluate\_model.py
    - models.py
    - pytorch\_unet.py
    - render.py
    - train.py
    - utils.py
- study them and identify a good start for studying and copying/adapting

- in fast-sr-unet/evaluate\_mode.py there is a ~260-lines-of-code function really messed up that can be useful as a reference to evaluate different metrics on low vs high quality videos
- start packaging the ./code/ folder in another repo
    - add tests/ and \<package-name\>/ folders to the new repo
    - manage it with poetry
    - poetry vs conda: understand which one would be the best to manage the venv for the project

- pytorch dataset

- look for LPIPS-Comp
- look for LPIPS
- look for SSIM / MS-SSIM

- binarize SR-UNet

---

## medium priority
- Professor suggests Comet.ml

---

## low priority
- replace fire with argparse

---

#### DONEs
- DONE - .gitignore literature/ folder, plus Python and LaTex stuff
- DONE - convert sh script for video preparation to Python
- DONE - implement a custom pytorch dataset

