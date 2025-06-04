# DiffuseSlide: Training-Free High Frame Rate Video Generation Diffusion

Official implementation of our paper:

> **DiffuseSlide: Training-Free High Frame Rate Video Generation Diffusion**  
> [Geunmin Hwang](https://github.com/GeunminHwang), [Hyun-kyu Ko](https://github.com/Ko-Lani), [Younghyun Kim](https://github.com/yhyun225), [Seungryong Lee](https://github.com/twowindragon), [Eunbyung Park](https://silverbottlep.github.io/)
> 📄 [arXiv 2506.01454](https://arxiv.org/abs/2506.01454) | 🌐 [Project Page](https://geunminhwang.github.io/DiffuseSlide/)

DiffuseSlide is a training-free framework for generating high frame-rate videos from pre-trained diffusion models.  
It leverages noise re-injection and sliding-window latent denoising to enhance temporal consistency and visual quality without additional fine-tuning.

---

## 🔧 Environment

### Python version
- Python 3.10.15

### CUDA version
- CUDA 11.8 **or** CUDA 12.1

### Required packages
- `torch==2.1.1`
- `diffusers==0.27.2`

---

## 📦 Environment Setup

### 1. Create conda environment
```shell
conda create -n DiffuseSlide python=3.10.15 -y
conda activate DiffuseSlide
```

### 2. Install packages

#### For CUDA 11.8:
```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchmetrics xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### For CUDA 12.1:
```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchmetrics xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🚀 Inference

### 4× Frame Rate Inference
```shell
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/4x_config.yaml
```

### 2× Frame Rate Inference
```shell
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/2x_config.yaml
```

---

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{hwang2025diffuseslide,
  title={DiffuseSlide: Training-Free High Frame Rate Video Generation Diffusion}, 
  author={Geunmin Hwang and Hyun-kyu Ko and Younghyun Kim and Seungryong Lee and Eunbyung Park},
  year={2025},
  eprint={2506.01454},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.01454}
}
```
