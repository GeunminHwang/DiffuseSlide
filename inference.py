import torch

import argparse
from models.DiffuseSlide import StableVideoDiffusion
from diffusers.utils import load_image, export_to_video
from omegaconf import OmegaConf

import os

from utils.make_result_folder import create_next_numbered_folder
from utils.utils import save_psnr_ssim_to_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
args = parser.parse_args()
config = OmegaConf.load(args.config)

device = "cuda" if torch.cuda.is_available() else "cpu"

DiffuseSlide_pipeline = StableVideoDiffusion(config, device)
DiffuseSlide_pipeline.to(device)

image_path = config.input_image_path
image = load_image(image_path)

output_path = create_next_numbered_folder(config.output_path)
frames, origin_frames, intepol_frames, psnr, ssim, execution_time = DiffuseSlide_pipeline(image)

if frames is not None:
    folder_path = os.path.join(output_path)
    os.makedirs(folder_path, exist_ok=True)
    
    # for 2x
    file_name = config.output_file_name + "_fps_14.mp4"
    result_path = os.path.join(folder_path, file_name)
    export_to_video(frames[0], result_path, fps=14)
    
    # for 4x
    file_name = config.output_file_name + "_fps_28.mp4"
    result_path = os.path.join(folder_path, file_name)
    export_to_video(frames[0], result_path, fps=28)
    
if origin_frames is not None:
    result_path = os.path.join(folder_path, "origin.mp4")
    export_to_video(origin_frames[0], result_path, fps=7)

if intepol_frames is not None:
    
    # for 2x
    result_path = os.path.join(folder_path, f"{config.output_file_name}_only_interpol.mp4")
    export_to_video(intepol_frames[0], result_path, fps=14)
    
    # for 4x
    result_path = os.path.join(folder_path, f"{config.output_file_name}_only_interpol_fps_28.mp4")
    export_to_video(intepol_frames[0], result_path, fps=28)

if psnr is not None and ssim is not None and execution_time is not None:
    file_name = 'metric.json'
    result_path = os.path.join(folder_path, file_name)
    save_psnr_ssim_to_json(psnr.item(), ssim.item(), execution_time, result_path)


if psnr is not None and ssim is not None and execution_time is not None:
    file_name = 'metric.json'
    result_path = os.path.join(output_path, file_name)
    save_psnr_ssim_to_json(psnr.item(), ssim.item(), execution_time, result_path)

result_path = os.path.join(output_path, "config.yaml")
OmegaConf.save(config, result_path)