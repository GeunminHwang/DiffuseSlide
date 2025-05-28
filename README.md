# DiffuseSlide: Training-Free High Frame Rate Video Generation Diffusion

## Environment
#### Python 3.10.15
#### cuda 11.8 or cuda 12.1
#### Packages:
* torch==2.1.1   
* diffuesrs==0.27.2

## Environment Setting
First create the anaconda environment.
```Shell
conda create -n DiffuseSlide python=3.10.15 -y
conda activate DiffuseSlide
```

Install the required packages. (cuda 11.8)
```Shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchmetrics xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
Install the required packages. (cuda 12.1)
```Shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchmetrics xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Inference 
### 4x Inference
```Shell
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/4x_config.yaml
```
### 2x Inference
```Shell
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/2x_config.yaml
```
