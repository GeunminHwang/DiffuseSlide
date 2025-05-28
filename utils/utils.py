import torch
import torch.fft as fft
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import math
import os
import json
from tqdm import tqdm
import numpy as np
from scipy.interpolate import CubicSpline

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def freq_mix_3d_latents(origin, sample, LPF):
    origin_freq = fft.fftn(origin, dim=(-3, -2, -1))
    origin_freq = fft.fftshift(origin_freq, dim=(-3, -2, -1))
    sample_freq = fft.fftn(sample, dim=(-3, -2, -1))
    sample_freq = fft.fftshift(sample_freq, dim=(-3, -2, -1))
    
    HPF = 1 - LPF
    sample_freq_low = sample_freq * HPF
    origin_freq_high = origin_freq * LPF
    sample_freq_mixed = sample_freq_low + origin_freq_high
    
    sample_freq_mixed = fft.ifftshift(sample_freq_mixed, dim=(-3, -2, -1))
    sample_mixed = fft.ifftn(sample_freq_mixed, dim=(-3, -2, -1)).real
    
    return sample_mixed

def get_freq_filter_3D(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask

def calculate_psnr(video1, video2, data_range=1.0):
    """
    Calculate PSNR for batched videos.
    video1, video2: torch tensor of shape (batch, frame_nums, channels, height, width)
    data_range: The maximum possible pixel value of the image (1.0 if the video is normalized [0, 1]).
    """
    # Calculate PSNR frame by frame
    batch_size, frame_nums, _, _, _ = video1.shape
    psnr_values = []
    
    for b in range(batch_size):
        for f in range(frame_nums):
            psnr_value = peak_signal_noise_ratio(video1[b, f], video2[b, f], data_range=data_range)
            psnr_values.append(psnr_value)
    
    # Convert to tensor and take mean
    psnr_values = torch.stack(psnr_values)
    return psnr_values.mean()

def calculate_ssim(video1, video2, data_range=1.0):
    """
    Calculate SSIM for batched videos.
    video1, video2: torch tensor of shape (batch, frame_nums, channels, height, width)
    data_range: The maximum possible pixel value of the image (1.0 if the video is normalized [0, 1]).
    """
    batch_size, frame_nums, _, _, _ = video1.shape
    ssim_values = []
    
    for b in range(batch_size):
        for f in range(frame_nums):
            v1 = video1[b, f].unsqueeze(0) if len(video1[b, f].shape) == 3 else video1[b, f]
            v2 = video2[b, f].unsqueeze(0) if len(video2[b, f].shape) == 3 else video2[b, f]            
            
            ssim_value = structural_similarity_index_measure(v1, v2, data_range=data_range)
            ssim_values.append(ssim_value)

    # Convert to tensor and take mean
    ssim_values = torch.stack(ssim_values)
    return ssim_values.mean()

def save_psnr_ssim_to_json(psnr, ssim, execution_time, filepath):
    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    data = {
        "PSNR": psnr,
        "SSIM": ssim,
        "Excution Time(sec)": execution_time
    }
    
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def get_all_first_frame_paths(source_path):
    video_folders = sorted(os.listdir(source_path))  # 모든 video 폴더를 정렬된 순서로 가져옴
    first_frame_paths = []
    
    for video_folder in tqdm(video_folders):  # 모든 video 폴더에 대해 반복
        frame_path = os.path.join(source_path, video_folder, 'frame1.jpg')  # 각 video 폴더의 첫번째 frame 경로
        # if os.path.exists(frame_path):  # 경로가 존재하는지 확인
        #     first_frame_paths.append(frame_path)
        # else:
        #     print(f"Warning: {frame_path} does not exist.")  # 경로가 없을 경우 경고 출력
        first_frame_paths.append(frame_path)
    return first_frame_paths

def path_filter(first_frame_paths, output_path):
    filtered_paths = []

    for path in first_frame_paths:
        # video_num 추출
        video_num = path.split('/')[-2]
        
        # output_path에 video_num 디렉토리가 있는지 확인
        if not os.path.exists(os.path.join(output_path, video_num)):
            filtered_paths.append(path)
    
    return filtered_paths
        
def gaussian_kernel1d(kernel_size: int, sigma: float):
    """1D Gaussian kernel 생성"""
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def apply_temporal_gaussian_smoothing(video: torch.Tensor, kernel_size: int, sigma: float):
    """
    Gaussian temporal smoothing을 적용하는 함수.
    
    Parameters:
    - video: (B, T, C, H, W) shape의 비디오 텐서 (B는 배치 크기, T는 프레임 수, C는 채널 수, H는 높이, W는 너비)
    - kernel_size: Gaussian 커널 크기
    - sigma: Gaussian 커널의 표준 편차
    
    Returns:
    - smoothed_video: Gaussian smoothing이 적용된 비디오 텐서
    """
    B, T, C, H, W = video.shape
    
    # 1D Gaussian kernel 생성
    kernel = gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1).to(video.device)  # (1, 1, kernel_size)
    
    # 비디오의 시간 축에 padding 추가
    pad_size = kernel_size // 2
    padded_video = F.pad(video, (0, 0, 0, 0, 0, 0, pad_size, pad_size), mode='reflect')  # (B, T + 2*pad_size, C, H, W)
    
    # (B, C, H, W, T + padding) 형태로 변환 후, 시간 축에서 conv1d 적용
    padded_video = padded_video.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T) -> (B, C, H, W, T + padding)
    
    # 시간 축에 대해 Gaussian smoothing 적용 (Conv1d 적용을 위해 마지막 차원을 T로 맞춰줌)
    smoothed_video = F.conv1d(
        padded_video.view(B * C * H * W, 1, T + 2 * pad_size),  # (B*C*H*W, 1, T + padding)
        kernel, 
        groups=B * C * H * W
    ).view(B, C, H, W, T)  # 원래 차원으로 다시 변환
    
    # 다시 (B, T, C, H, W)로 차원 변환
    smoothed_video = smoothed_video.permute(0, 4, 1, 2, 3)  # (B, C, H, W, T) -> (B, T, C, H, W)

    return smoothed_video

def cubic_spline_interpolation(frame_a, frame_b):
    # Move the tensors to CPU first since scipy.interpolate works on numpy arrays
    frame_a_np = frame_a.cpu().numpy()
    frame_b_np = frame_b.cpu().numpy()
    
    # Create a cubic spline for each pixel (spline interpolation for frames a and b)
    x = np.array([0, 1])  # Known keyframe positions
    cs = CubicSpline(x, np.stack([frame_a_np, frame_b_np], axis=0))
    
    # Interpolated frame at position 0.5 (midway between the two keyframes)
    interpolated_frame_np = cs(0.5)
    
    # Convert back to torch tensor and move it to GPU
    return torch.tensor(interpolated_frame_np).to(frame_a.device)

def temporal_pca(latents, n_components=2):
    latents = latents.to(dtype=torch.float32)

    # latents shape: (batch, frame_num, channel, height, width)
    batch_size, frame_num, channels, height, width = latents.shape
    
    # Flatten spatial and channel dimensions (for temporal PCA)
    latents_reshaped = latents.permute(0, 2, 3, 4, 1).reshape(batch_size * channels * height * width, frame_num)
    
    # Mean center the data
    latents_mean = latents_reshaped.mean(dim=1, keepdim=True)
    latents_centered = latents_reshaped - latents_mean
    
    # Perform SVD for PCA (differentiable)
    U, S, Vt = torch.svd(latents_centered)
    
    # Select the top 'n_components' components
    Vt = Vt[:, :n_components]  # shape: (frame_num, n_components)
    
    # Project original data onto principal components
    projected = torch.matmul(latents_centered, Vt)  # shape: (batch * channels * height * width, n_components)
    
    # Reshape back to (batch, channel, height, width, n_components)
    projected = projected.view(batch_size, channels, height, width, n_components).permute(0, 4, 1, 2, 3)
    
    return projected

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    return generator