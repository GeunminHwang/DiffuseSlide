import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
import PIL.Image
import logging
import random

import torch
import torch.nn.functional as F

from diffusers import EulerDiscreteScheduler, StableVideoDiffusionPipeline
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
import PIL.Image
import logging

import torch
import torch.nn.functional as F

from diffusers import EulerDiscreteScheduler, StableVideoDiffusionPipeline

from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video

from utils.utils import (
    get_freq_filter_3D, 
    freq_mix_3d,
    calculate_psnr,
    calculate_ssim,
    set_seeds
)

from utils.interpolation import (
    slerp,
)

from utils.guide import MSEGuidance

logger = logging.getLogger()

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs

class StableVideoDiffusion(object):
    def __init__(self, config, device, use_amp=True):
        
        self.cfg = config.diffusion
        self.fft_cfg = config.fft
        self.sliding_windows_cfg = config.sliding_windows
        self.weights_dtype = torch.float16 if use_amp else torch.float32
        self.configure(config)
        self.generator = set_seeds(self.cfg.seed)
        self.fps = [fps - 1 for fps in self.cfg.fps]
            
        self.cond_fn = MSEGuidance(
            scale=self.cfg.g_scale, 
            t_start=0, 
            t_stop=0,
            space='latent', 
            repeat=1
            )
        
    def configure(self, config):
        # load Stable Diffusion
        logger.info("Loading Stable Diffusion...")
        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": "./cache",
        }
        
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs
        )

        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

        self.pipeline = pipeline

        logger.info("Loaded Stable Video Diffusion!")

        for p in self.pipeline.vae.parameters():
            p.requires_grad_(False)
        for p in self.pipeline.unet.parameters():
            p.requires_grad_(False)
        
        self.needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
    
    def to(self, args):
        self.pipeline.to(args)
        self.device = self.pipeline.device
        if self.cfg.model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        
        if self.cfg.torch_compile_unet:
            self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
            
    def edm_scheduler_step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = 0.0,
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_pred_original_sample: bool = False,
    ):  
        assert isinstance(self.pipeline.scheduler, EulerDiscreteScheduler)
        config = self.pipeline.scheduler.config

        self.pipeline.scheduler._init_step_index(timestep)        
        step_index = self.pipeline.scheduler.step_index
        
        sigma = self.pipeline.scheduler.sigmas[step_index]
        gamma = min(s_churn / (len(self.pipeline.scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if config.prediction_type == "original_sample" or config.prediction_type == "sample":
            pred_original_sample = model_output
        elif config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
                
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        
        dt = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        if return_pred_original_sample:
            return (prev_sample, pred_original_sample)
        
        return (prev_sample, )   
    
    def edm_scheduler_step_optimize(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = 0.0,
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_pred_original_sample: bool = False,
    ):  
        assert isinstance(self.pipeline.scheduler, EulerDiscreteScheduler)
        config = self.pipeline.scheduler.config

        self.pipeline.scheduler._init_step_index(timestep)        
        step_index = self.pipeline.scheduler.step_index
        
        sigma = self.pipeline.scheduler.sigmas[step_index]
        gamma = min(s_churn / (len(self.pipeline.scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if config.prediction_type == "original_sample" or config.prediction_type == "sample":
            pred_original_sample = model_output
        elif config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
                
        # 2. Convert to an ODE derivative
        indices = torch.arange(0, pred_original_sample.size(1), step=self.cfg.num_interpolations + 1)
        pred_z0_origin = pred_original_sample[:, indices, :, :, :]
   
        # grad_rescale = sigma_hat / (self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index + 1] - sigma_hat)
        grad_rescale = 1
        # apply guidance for multiple times
        loss_vals = []
        
        for _ in range(self.cond_fn.repeat):
            # set target and pred for gradient computation
            target, pred = None, None
            
            if self.cond_fn.space == "latent":
                target = self.cond_fn.target.half()
                pred = pred_z0_origin
                    
            elif self.cond_fn.space == "rgb":
                # We need to backward gradient to x0 in latent space, so it's required
                # to trace the computation graph while decoding the latent.
                with torch.enable_grad():
                    target = self.cond_fn.target
                    pred_z0_rg = pred_z0_origin.detach().clone().requires_grad_(True)
                    pred = self.pipeline.decode_latents(pred_z0_rg, pred_z0_rg.shape[1], self.cfg.decode_chunk_size)
                    assert pred.requires_grad
            
            else:
                raise NotImplementedError(self.cond_fn.space)
            
            # compute gradient
            delta_pred, loss_val = self.cond_fn(target, pred, 0)
            loss_vals.append(loss_val)
            
            # update pred_x0 w.r.t gradient
            if self.cond_fn.space == "latent":
                delta_pred_x0 = delta_pred
                pred_z0_origin = pred_z0_origin + delta_pred_x0 * grad_rescale
                    
            elif self.cond_fn.space == "rgb":
                pred.backward(delta_pred)
                delta_pred_x0 = pred_z0_rg.grad
                pred_z0_origin = pred_z0_origin + delta_pred_x0 * grad_rescale
            
            else:
                raise NotImplementedError(self.cond_fn.space)
        
        pred_original_sample[:, indices, :, :, :] = pred_z0_origin
        
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index + 1] - sigma_hat
        prev_sample = sample + derivative * dt

        if return_pred_original_sample:
            return (prev_sample, pred_original_sample)
        
        return (prev_sample, )   
        
    def scheduler_add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.pipeline.scheduler.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.pipeline.scheduler.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.pipeline.scheduler.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.pipeline.scheduler.begin_index is None:
            step_indices = [self.pipeline.scheduler.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.pipeline.scheduler.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.pipeline.scheduler.step_index] * timesteps.shape[0]
        else:
            # add noise is called bevore first denoising step to create inital latent(img2img)
            step_indices = [self.pipeline.scheduler.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples, sigma
    
    def interpolate_frame_latent(
        self,
        latents,
        num_interpolations
    ):
        batch_size, frame_nums, channels, height, width = latents.shape
        num_interpolations = num_interpolations
        new_frame_nums = frame_nums * (1 + num_interpolations)  
        interpolated_latents = torch.zeros(batch_size, new_frame_nums, channels, height, width, dtype=latents.dtype, device=latents.device)

        for i in range(frame_nums - 1):
            interpolated_latents[:, i * (num_interpolations + 1), :, :, :] = latents[:, i, :, :, :]
            
            for j in range(1, num_interpolations + 1):
                alpha = j / (num_interpolations + 1) 
                
                # slerp interpolation
                interpolated_latents[:, i * (num_interpolations + 1) + j, :, :, :] = \
                    slerp(alpha, latents[:, i, :, :, :], latents[:, i + 1, :, :, :])
                
                # linear interpolation
                # interpolated_latents[:, i * (num_interpolations + 1) + j, :, :, :] = \
                #     (1 - alpha) * latents[:, i, :, :, :] + alpha * latents[:, i + 1, :, :, :]
                    
        for i in range(num_interpolations + 1): 
            interpolated_latents[:, -(i+1), :, :, :] = latents[:, -1, :, :, :]
        
        return interpolated_latents
    
    def encode_image_vae(self, image, num_frames ,height, width):
        image = self.pipeline.image_processor.preprocess(image, height=height, width=width).to(self.device)
        noise = randn_tensor(image.shape, generator=self.generator, device=self.device, dtype=image.dtype)
        image = image + self.cfg.noise_aug_strength * noise
        
        if self.needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float32)
        
        with torch.no_grad():
            image_latents = self.pipeline._encode_vae_image(
                image,
                device=self.device,
                num_videos_per_prompt=self.cfg.num_videos_per_prompt,
                do_classifier_free_guidance=self.cfg.do_classifier_free_guidance,
            )
        
        image_latents = image_latents.to(self.image_embeddings.dtype)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        if self.needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float16)
            
        return image_latents
    
    def get_add_time_ids(self, flag):
        added_time_ids = self.pipeline._get_add_time_ids(
            self.fps[flag],
            self.cfg.motion_bucket_id[flag],
            self.cfg.noise_aug_strength,
            self.image_embeddings.dtype,
            self.cfg.batch_size,
            self.cfg.num_videos_per_prompt,
            self.cfg.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(self.device)
        return added_time_ids
    
    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.pipeline.vae_scale_factor,
            width // self.pipeline.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            init_noise = latents.detach().clone()
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        return latents, init_noise
    
    def prepare_guidance_scale(self, latents, num_frames, i):
        guidance_scale = torch.linspace(self.cfg.min_guidance_scale[i], self.cfg.max_guidance_scale[i], num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(self.device, latents.dtype)
        guidance_scale = guidance_scale.repeat(self.cfg.batch_size * self.cfg.num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        
        self._guidance_scale = guidance_scale
        
        return guidance_scale
    
    def add_noise(self, latents, timesteps):
        noise = torch.randn_like(latents)
        start_t = torch.tensor([timesteps[0].item()]).to(timesteps[0].device)
        latents = self.pipeline.scheduler.add_noise(latents, noise, start_t)
        
        return latents
    
    def noise_swap(self, latents, init_noise, start_T):
        epsilon = torch.randn_like(latents)
        
        z_t, sigma = self.scheduler_add_noise(latents, init_noise, start_T)
        epsilon_t = epsilon * sigma
        
        mask = get_freq_filter_3D(
            shape=z_t.permute(0, 2, 1, 3, 4).shape, 
            device=z_t.device,
            filter_type=self.fft_cfg.filter_type,
            n=self.fft_cfg.n,
            d_s=self.fft_cfg.d_s,
            d_t=self.fft_cfg.d_t
            )
        mask = mask.permute(0, 2, 1, 3, 4)
        
        z_t = freq_mix_3d(z_t.to(dtype=torch.float32), epsilon_t.to(dtype=torch.float32), mask)
        latents = z_t.to(dtype=torch.float16)
        
        new_epsion = latents / sigma
        
        return new_epsion
    
    def encode_temp_image_latents(self, temp_frames, index, kernel_size):
        
        temp_frames = tensor2vid(temp_frames[:,:,index:index+1,:,:], self.pipeline.image_processor, output_type="pil")
        
        image = []
        for frame in temp_frames:
            image.append(frame[0])
        
        self.image_embeddings = self.pipeline._encode_image(image, self.device, self.cfg.num_videos_per_prompt, self.cfg.do_classifier_free_guidance)   
        image_latents = self.encode_image_vae(image, kernel_size, self.cfg.height, self.cfg.width)
        return image_latents
    
    def denoising_diffusion_process(
        self,
        image_latents,
        latents,
        timesteps,
        guidance_scale,
    ):
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                
                # predict the noise residual
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=self.image_embeddings,
                    added_time_ids=self.added_time_ids,
                    return_dict=False,
                )[0]
                
                # perform guidance
                if self.cfg.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
                latents = self.edm_scheduler_step(noise_pred, t, latents, self.generator)[0]
                
                self.pipeline.scheduler._step_index += 1
                
        return latents    

    def denoising_diffusion_process_fuse(
        self,
        image_latents,
        latents,
        timesteps,
        guidance_scale,
        flag,
    ):
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                
                # predict the noise residual
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=self.image_embeddings,
                    added_time_ids=self.added_time_ids,
                    return_dict=False,
                )[0]
                
                # perform guidance
                if self.cfg.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
                latents = self.edm_scheduler_step(noise_pred, t, latents, self.generator)[0]
                
                # noise_re-injection
                if i < self.cfg.fuse_threshold[flag]:
                    sigma_t = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index]
                    sigma_tm1 = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index+1]
                    sigma = torch.sqrt(sigma_t**2-sigma_tm1**2)
                    
                    for j in range(self.cfg.fuse_step[flag]):
                        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
                        noise = noise * sigma
                        latents = latents + noise
                        
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                        latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                        # Concatenate image_latents over channels dimention
                        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                            
                        noise_pred = self.pipeline.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=self.image_embeddings,
                            added_time_ids=self.added_time_ids,
                            return_dict=False,
                        )[0]

                        if self.cfg.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        latents = self.edm_scheduler_step(noise_pred, t, latents, self.generator)[0]
                    
                self.pipeline.scheduler._step_index += 1
                    
        return latents

    def denoise_diffusion_process_long_slide(
        self,
        latents,
        timesteps,
        temp_frames,
        flag
    ):
        k = self.sliding_windows_cfg.kernel_size[flag]
        s = self.sliding_windows_cfg.stride[flag]
        step = int((latents.shape[1] - k) / s + 1)
        
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                overlap_noise = torch.zeros_like(latents)
                overlap_counter = torch.zeros_like(latents)
                
                for j in tqdm(range(step)):
                    image_latents = self.encode_temp_image_latents(temp_frames=temp_frames, index=j*s, kernel_size=k)
                    sliced_latents = latents[:, j * s : j * s + k, :, :, :]
                    guidance_scale = self.prepare_guidance_scale(sliced_latents, sliced_latents.shape[1], flag)
                    
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([sliced_latents] * 2) if self.cfg.do_classifier_free_guidance else sliced_latents
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                    
                    # predict the noise residual
                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=self.image_embeddings,
                        added_time_ids=self.added_time_ids,
                        return_dict=False,
                    )[0]
                    
                    # perform guidance
                    if self.cfg.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    overlap_noise[:, j*s:j*s+k, :, :, :] += noise_pred
                    overlap_counter[:, j*s:j*s+k, :, :, :] += 1
                
                overlap_noise = overlap_noise / overlap_counter
                
                latents = self.edm_scheduler_step_optimize(overlap_noise, t, latents, self.generator)[0]
                
                self.pipeline.scheduler._step_index += 1
        
        return latents
    
    def denoise_diffusion_process_long_slide_fuse(
        self,
        latents,
        timesteps,
        temp_frames,
        flag
    ):
        k = self.sliding_windows_cfg.kernel_size[flag]
        s = self.sliding_windows_cfg.stride[flag]
        if (latents.shape[1] - k) % s == 0:
            last_step_change = False
            step = int((latents.shape[1] - k) / s + 1)
        else:
            last_step_change = True 
            step = int((latents.shape[1] - k) / s + 1) + 1

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                overlap_noise = torch.zeros_like(latents)
                overlap_counter = torch.zeros_like(latents)
                
                for j in range(step):
                    if j == step - 1 and last_step_change:
                        last_s = self.cfg.num_interpolations + 1
                        last_k = latents.shape[1] - (int((latents.shape[1] - k) / s) * s + last_s)
                        image_latents = self.encode_temp_image_latents(temp_frames=temp_frames, index=(j - 1)*s + last_s, kernel_size=last_k)
                        sliced_latents = latents[:, (j - 1)*s+last_s : (j - 1)*s+last_s+last_k, :, :, :]
                    
                    else:     
                        image_latents = self.encode_temp_image_latents(temp_frames=temp_frames, index=j*s, kernel_size=k)
                        sliced_latents = latents[:, j * s : j * s + k, :, :, :]
                        
                    guidance_scale = self.prepare_guidance_scale(sliced_latents, sliced_latents.shape[1], flag)
                    
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([sliced_latents] * 2) if self.cfg.do_classifier_free_guidance else sliced_latents
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                    
                    # predict the noise residual
                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=self.image_embeddings,
                        added_time_ids=self.added_time_ids,
                        return_dict=False,
                    )[0]
                    
                    # perform guidance
                    if self.cfg.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                    if j == step - 1 and last_step_change:
                        overlap_noise[:, (j - 1)*s+last_s:(j - 1)*s+last_s+last_k, :, :, :] += noise_pred
                        overlap_counter[:, (j - 1)*s+last_s:(j - 1)*s+last_s+last_k, :, :, :] += 1
                    else:
                        overlap_noise[:, j*s:j*s+k, :, :, :] += noise_pred
                        overlap_counter[:, j*s:j*s+k, :, :, :] += 1
                
                overlap_noise = overlap_noise / overlap_counter

                latents = self.edm_scheduler_step_optimize(overlap_noise, t, latents, self.generator)[0]
                    
                if i < self.cfg.fuse_threshold[flag]:
                    sigma_t = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index]
                    sigma_tm1 = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.step_index+1]
                    sigma = torch.sqrt(sigma_t**2-sigma_tm1**2)
                    
                    for l in range(self.cfg.fuse_step[flag]):
                        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
                        noise = noise * sigma
                        latents = latents + noise
                        
                        overlap_noise = torch.zeros_like(latents)
                        overlap_counter = torch.zeros_like(latents)
                        
                        for j in range(step):
                            if j == step - 1 and last_step_change:
                                last_s = self.cfg.num_interpolations + 1
                                last_k = latents.shape[1] - (int((latents.shape[1] - k) / s) * s + last_s)
                                image_latents = self.encode_temp_image_latents(temp_frames=temp_frames, index=(j - 1)*s + last_s, kernel_size=last_k)
                                sliced_latents = latents[:, (j - 1)*s+last_s : (j - 1)*s+last_s+last_k, :, :, :]
                            else:     
                                image_latents = self.encode_temp_image_latents(temp_frames=temp_frames, index=j*s, kernel_size=k)
                                sliced_latents = latents[:, j * s : j * s + k, :, :, :]
                                
                            guidance_scale = self.prepare_guidance_scale(sliced_latents, sliced_latents.shape[1], flag)
                            
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([sliced_latents] * 2) if self.cfg.do_classifier_free_guidance else sliced_latents
                            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                            # Concatenate image_latents over channels dimention
                            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                            
                            # predict the noise residual
                            noise_pred = self.pipeline.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=self.image_embeddings,
                                added_time_ids=self.added_time_ids,
                                return_dict=False,
                            )[0]
                            
                            # perform guidance
                            if self.cfg.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                                
                            if j == step - 1 and last_step_change:
                                overlap_noise[:, (j - 1)*s+last_s:(j - 1)*s+last_s+last_k, :, :, :] += noise_pred
                                overlap_counter[:, (j - 1)*s+last_s:(j - 1)*s+last_s+last_k, :, :, :] += 1
                            else:
                                overlap_noise[:, j*s:j*s+k, :, :, :] += noise_pred
                                overlap_counter[:, j*s:j*s+k, :, :, :] += 1                            
                                
                        overlap_noise = overlap_noise / overlap_counter
                        latents = self.edm_scheduler_step_optimize(overlap_noise, t, latents, self.generator)[0]
                        
                self.pipeline.scheduler._step_index += 1
        
        return latents
    
    @torch.no_grad()
    def __call__(self, image):
        import time
        start_time = time.time()
        height, width = self.cfg.height, self.cfg.width
        denoise_start, denoise_end = self.cfg.denoise_start, self.cfg.denoise_end

        if isinstance(image, PIL.Image.Image):
            self.cfg.batch_size = 1
        elif isinstance(image, list):
            self.cfg.batch_size = len(image)
        else:
            self.cfg.batch_size = image.shape[0]
        
        for flag, denoise_steps in enumerate(zip(denoise_start, denoise_end)):
            # 1. Encode input image for image embedding
            self.image_embeddings = self.pipeline._encode_image(image, self.device, self.cfg.num_videos_per_prompt, self.cfg.do_classifier_free_guidance)
            
            # 2. Get Added time ids
            self.added_time_ids = self.get_add_time_ids(flag)
            start, end = denoise_steps
            
            # 3. Prepare timesteps
            if flag == 0:
                self.pipeline.scheduler.set_timesteps(self.cfg.num_inference_steps, device=self.device)
                timesteps = self.pipeline.scheduler.timesteps
                timesteps = timesteps[start:end]
            
            # 4. Prepare latent variables
            # interpolation
            if flag != 0:
                if self.cfg.interpolation_enable[flag]:
                    latents = self.interpolate_frame_latent(latents, self.cfg.num_interpolations)
                    
                    if self.cfg.save_interpol:
                        interpol_latents = latents
                        interpol_frames = None
                        interpol_frames = self.pipeline.decode_latents(interpol_latents, interpol_latents.shape[1], self.cfg.decode_chunk_size)
                        interpol_frames = tensor2vid(interpol_frames, self.pipeline.image_processor, output_type="np")
                else:
                    latents = latents
                
            # only noisy latents
            else:
                latents = torch.FloatTensor = None
                num_channels_latents = self.pipeline.unet.config.in_channels
                z_T, _ = self.prepare_latents(
                    self.cfg.batch_size * self.cfg.num_videos_per_prompt,
                    self.cfg.num_frames,
                    num_channels_latents,
                    height,
                    width,
                    self.image_embeddings.dtype,
                    self.device,
                    self.generator,
                    latents
                    )

                epsilon = torch.randn_like(z_T)
                epsilon_T = epsilon * self.pipeline.scheduler.init_noise_sigma
                mask = get_freq_filter_3D(
                    shape=z_T.permute(0, 2, 1, 3, 4).shape, 
                    device=z_T.device,
                    filter_type=self.fft_cfg.filter_type,
                    n=self.fft_cfg.n,
                    d_s=self.fft_cfg.d_s,
                    d_t=self.fft_cfg.d_t
                    )
                mask = mask.permute(0, 2, 1, 3, 4)
                
                z_T = freq_mix_3d(z_T.to(dtype=torch.float32),
                                epsilon_T.to(dtype=torch.float32), 
                                mask)
                
                latents = z_T.to(dtype=torch.float16)
                
            # 5. Prepare guidance scale
            if self.cfg.split_enable[flag] == False:
                guidance_scale = self.prepare_guidance_scale(latents, latents.shape[1], flag)

            # 6. Encode input image usinag vae
            image_latents = self.encode_image_vae(image, latents.shape[1], height, width)
            
            # 7. denoise the video, get clean video prediction
            if flag != 0:
                temp_frames = self.pipeline.decode_latents(latents, latents.shape[1], self.cfg.decode_chunk_size)
                self.pipeline.scheduler.set_timesteps(self.cfg.num_inference_steps, device=self.device)
                timesteps = self.pipeline.scheduler.timesteps
                timesteps = timesteps[start:end]
                
                latents = self.add_noise(latents=latents, timesteps=timesteps)
                
                if self.cfg.fuse_enable[flag]:
                    latents = self.denoise_diffusion_process_long_slide_fuse(
                        latents=latents,
                        timesteps=timesteps,
                        temp_frames=temp_frames,
                        flag=flag
                    )
                
                else:
                    latents = self.denoise_diffusion_process_long_slide(
                        latents=latents,
                        timesteps=timesteps,
                        temp_frames=temp_frames,
                        flag=flag
                    )

            else:
                latents = self.denoising_diffusion_process(
                    image_latents=image_latents, 
                    latents=latents, 
                    timesteps=timesteps, 
                    guidance_scale=guidance_scale
                    )
                
                origin_latents = latents
                origin_frames = None
                
                self.cond_fn.load_target(origin_latents)

                if self.cfg.save_origin:                
                    origin_frames = self.pipeline.decode_latents(origin_latents, origin_latents.shape[1], self.cfg.decode_chunk_size)

                    if self.cfg.metrics:        
                        if any(self.cfg.interpolation_enable) & self.cfg.metrics:
                            origin_frames_pt = tensor2vid(origin_frames, self.pipeline.image_processor, output_type="pt")
                            
                    origin_frames = tensor2vid(origin_frames, self.pipeline.image_processor, output_type="np")
                    
            if self.needs_upcasting:
                self.pipeline.vae.to(dtype=torch.float16)
            
        if self.needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float16)
        frames = self.pipeline.decode_latents(latents, latents.shape[1], self.cfg.decode_chunk_size)
            
        # frames: batch, channels, frame_nums, width, height
        if any(self.cfg.interpolation_enable) & self.cfg.metrics:
            frames_pt = tensor2vid(frames, self.pipeline.image_processor, output_type="pt")
            
            multiple = frames_pt.shape[1] // self.cfg.num_frames
            psnr = calculate_psnr(origin_frames_pt, frames_pt[:, 0::multiple, :, :, :])
            ssim = calculate_ssim(origin_frames_pt, frames_pt[:, 0::multiple, :, :, :])
            
            print("PSNR:", psnr.item())
            print("SSIM:", ssim.item())
        
        else:
            psnr = None
            ssim = None
        
        frames = tensor2vid(frames, self.pipeline.image_processor, output_type="np")   
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Excution Time: {execution_time} sec")
        
        return frames, origin_frames, interpol_frames, psnr, ssim, execution_time