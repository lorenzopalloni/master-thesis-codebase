# Master's thesis codebase

### To compile torchvision with ffmpeg support:
```bash
git clone git@github.com:pytorch/vision.git
sudo apt install ffmpeg
sudo apt install libavcodec-dev
sudo apt install libavfilter-dev
sudo apt install libswscale-dev
pip install --upgrade av==8.1.0
cd ./vision && python3.8 setup.py install
```

### To compile ffmpeg on Ubuntu:
- https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
