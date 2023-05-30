# Master's thesis codebase
Codebase for my Master's Thesis ["OPTIMIZATION TECHNIQUES OF DEEP LEARNING MODELS FOR VISUAL QUALITY IMPROVEMENT"](https://github.com/lorenzopalloni/master-thesis).

### Abstract
This thesis examines the efficacy of quantization techniques for enhancing the inference speed and reducing the memory usage of deep learning models applied to video restoration tasks. The research investigates the implementation and evaluation of post-training quantization using TensorRT, an NVIDIA tool for inference optimization. The results indicate that reducing the precision of weights and activations substantially decreases computational complexity and memory requirements without compromising performance. In particular, the INT8-optimized UNet and SRUNet models achieve 2.38X and 2.26X speedup compared to their plain implementations, respectively, while also achieving memory consumption reductions of 63.3\% for UNet and 53.8\% for SRUNet. These findings should contribute to the development of more practical and efficient video restoration models for real-world applications.

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
