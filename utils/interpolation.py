import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torch.nn.functional as F
from torch.nn.functional import grid_sample

def slerp(val, low, high):
    """Spherical linear interpolation for 4D tensors (batch, channel, height, width)."""
    # Reshape to (batch, -1) to compute the norm over (channel, height, width)
    low_flat = low.view(low.shape[0], -1)
    high_flat = high.view(high.shape[0], -1)

    # Normalize the flattened tensors along dimension 1
    low_norm = torch.norm(low_flat, dim=1, keepdim=True)
    high_norm = torch.norm(high_flat, dim=1, keepdim=True)

    low_normalized = low_flat / low_norm
    high_normalized = high_flat / high_norm

    # Compute the dot product
    dot_product = torch.sum(low_normalized * high_normalized, dim=1, keepdim=True)
    omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    so = torch.sin(omega)

    # Handle cases where omega is 0 to avoid division by zero
    so = torch.where(so == 0, torch.tensor(1.0, device=so.device), so)

    # Compute the interpolated latent
    interp = (torch.sin((1.0 - val) * omega) / so) * low_flat + (torch.sin(val * omega) / so) * high_flat

    # Reshape back to the original (batch, channel, height, width)
    return interp.view_as(low)


def warp_frame(frame, flow, scaling_factor=1.0):
    """
    Warp an input frame using the given flow with a scaling factor.
    
    Args:
        frame: Tensor of shape (batch, channels, height, width)
        flow: Tensor of shape (batch, 2, height, width) (x, y flow)
        scaling_factor: Factor to scale the optical flow for interpolation.
        
    Returns:
        Warped frame tensor of shape (batch, channels, height, width)
    """
    B, C, H, W = frame.size()

    # Create a meshgrid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid_x = grid_x.to(frame.device).float()
    grid_y = grid_y.to(frame.device).float()

    # Normalize the grid to [-1, 1] for bilinear sampling
    norm_grid_x = (2.0 * grid_x / (W - 1)) - 1.0
    norm_grid_y = (2.0 * grid_y / (H - 1)) - 1.0
    grid = torch.stack((norm_grid_x, norm_grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

    # Get the flow for x and y directions and scale the flow
    flow_x, flow_y = flow[:, 0, :, :] * scaling_factor, flow[:, 1, :, :] * scaling_factor

    # Add the scaled flow to the grid coordinates
    warped_grid_x = norm_grid_x.unsqueeze(0).expand_as(flow_x) + (flow_x / W)
    warped_grid_y = norm_grid_y.unsqueeze(0).expand_as(flow_y) + (flow_y / H)

    # Stack and reshape the grid for sampling
    warped_grid = torch.stack((warped_grid_x, warped_grid_y), dim=3)
    
    # Apply the grid sample using the calculated warp grid
    warped_frame = F.grid_sample(frame, warped_grid, mode='bilinear', padding_mode='border')
    
    return warped_frame

def optical_flow_interpolation(frames, alpha=0.5):
    
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(frames.device)
    model = model.eval()
    
    batch_size, channels, num_frames, height, width = frames.shape
    interpolated_frames = torch.zeros(batch_size, channels, 2 * num_frames, height, width).to(frames.device)
    
    for i in range(num_frames - 1):
        # Extract two consecutive frames
        frame1 = frames[:, :, i, :, :]  # Add frame dimension
        frame2 = frames[:, :, i + 1, :, :]

        # RAFT forward pass to compute optical flow
        with torch.no_grad():
            flow_predictions = model(frame1, frame2)
            flow_fwd = flow_predictions[-1]  # Use the last flow prediction
            flow_predictions = model(frame2, frame1)
            flow_bwd = flow_predictions[-1]  # Use the last flow prediction

            
        # Warp frame1 towards frame2 using the computed flow
        warped_frame1 = warp_frame(frame1, flow_fwd, alpha)
        warped_frame2 = warp_frame(frame2, flow_bwd, alpha)

        # Insert original frame and interpolated frame into the output tensor
        # interpolated_frames[:, :, 2 * t, :, :] = frame1  # Original frame
        # interpolated_frames[:, :, 2 * t + 1, :, :] = warped_frame  # Interpolated frame
        interpolated_frames[:, :, 2 * i, :, :] = frame1# Original frame
        interpolated_frames[:, :, 2 * i + 1, :, :] = 0.5 * frame1 + 0.5 * warped_frame1 # Interpolated frame
        # interpolated_frames[:, :, 2 * i + 1, :, :] =  0.5 * warped_frame1 + 0.5 * warped_frame2 # Interpolated frame

    # Add the last frame (as no flow is computed for the final frame)
    interpolated_frames[:, :, -2, :, :] = frames[:, :, -1, :, :]
    warped_frame1 = warp_frame(frames[:, :, -1, :, :], flow_fwd, alpha)
    interpolated_frames[:, :, -1, :, :] = 0.5 * frames[:, :, -1, :, :] + 0.5 * warped_frame1
    
    del model
    torch.cuda.empty_cache()
    
    return interpolated_frames

