import torch
import math

def get_dct_matrix(N, dtype=torch.float32, device='cpu'):
    """
    Generate an N x N orthonormal DCT-II matrix.
    """
    n = torch.arange(N, dtype=dtype, device=device)
    k = n.view(-1, 1)
    # DCT-II formula
    dct = torch.cos(math.pi / N * (n + 0.5) * k)
    # Orthogonal normalization
    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / N)
    return dct

def dct2(x):
    """
    Perform 2D DCT on the last two dimensions of x.
    x shape: [B, C, H, W]
    """
    H, W = x.shape[-2], x.shape[-1]
    M_h = get_dct_matrix(H, dtype=x.dtype, device=x.device)
    M_w = get_dct_matrix(W, dtype=x.dtype, device=x.device)
    
    # 2D DCT: M_h @ x @ M_w.T
    out = torch.matmul(M_h, x)
    out = torch.matmul(out, M_w.T)
    return out

def idct2(X):
    """
    Perform 2D inverse DCT on the last two dimensions of X.
    X shape: [B, C, H, W]
    """
    H, W = X.shape[-2], X.shape[-1]
    M_h = get_dct_matrix(H, dtype=X.dtype, device=X.device)
    M_w = get_dct_matrix(W, dtype=X.dtype, device=X.device)
    
    # 2D IDCT: M_h.T @ X @ M_w
    out = torch.matmul(M_h.T, X)
    out = torch.matmul(out, M_w)
    return out

def make_frequency_mask(orig_h, orig_w, target_h, target_w, taper, device):
    """
    Create a 2D mask for blending low-frequency and high-frequency DCT coefficients.
    Mask value is 1.0 at low frequencies, tapering to 0.0 near the boundary of the original size.
    """
    mask_h = torch.zeros(target_h, device=device, dtype=torch.float32)
    mask_w = torch.zeros(target_w, device=device, dtype=torch.float32)
    
    if taper <= 0:
        mask_h[:orig_h] = 1.0
        mask_w[:orig_w] = 1.0
    else:
        # taper cannot exceed the original dimensions
        taper_h = min(taper, orig_h)
        taper_w = min(taper, orig_w)
        
        # 1.0 region
        mask_h[:orig_h - taper_h] = 1.0
        mask_w[:orig_w - taper_w] = 1.0
        
        # Taper region (cosine drop-off from 1 to 0)
        # Using 0.5 * (1 + cos(pi * x / taper))
        for i in range(taper_h):
            pos = i / taper_h
            mask_h[orig_h - taper_h + i] = 0.5 * (1.0 + math.cos(math.pi * pos))
            
        for i in range(taper_w):
            pos = i / taper_w
            mask_w[orig_w - taper_w + i] = 0.5 * (1.0 + math.cos(math.pi * pos))
            
    # 2D mask is outer product
    mask = torch.outer(mask_h, mask_w)
    # Shape: [1, 1, target_h, target_w] for broadcasting
    return mask.unsqueeze(0).unsqueeze(0)

class LatentSpectralExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", ),
                "scale_factor": ("FLOAT", {"default": 1.25, "min": 1.0, "max": 10.0, "step": 0.05}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "taper": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "blend_mode": (["variance_preserving", "linear", "hard"], {"default": "variance_preserving"}),
                "edm_style": ("BOOLEAN", {"default": True, "tooltip": "Convert sigma to t ∈ [0, 1] and back (highly recommended for SD1.5, SDXL, Karras schedules where sigma_max > 1)"})
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "sigma_aligned")
    FUNCTION = "expand"
    CATEGORY = "latent/spectral"

    def expand(self, latent, scale_factor, sigma, noise_strength, seed, taper, blend_mode, edm_style):
        samples = latent["samples"]
        orig_dtype = samples.dtype
        device = samples.device
        
        is_5d = (samples.ndim == 5)
        if is_5d:
            B, C, T, H, W = samples.shape
            # Reshape [B, C, T, H, W] -> [B * T, C, H, W]
            samples_4d = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B, C, H, W = samples.shape
            T = None
            samples_4d = samples
        
        # Calculate target latent dimensions, rounding to the nearest multiple of 8
        target_latent_height = max(H, int(round((H * scale_factor) / 8.0) * 8))
        target_latent_width = max(W, int(round((W * scale_factor) / 8.0) * 8))
        
        if target_latent_height <= H and target_latent_width <= W:
            # If target is not strictly larger, just return the original latent
            return (latent, sigma)

        # Compute DCT on original spatial dimension
        x32 = samples_4d.to(torch.float32)
        F_low = dct2(x32)
        
        # Prepare larger frequency canvas
        F_embed = torch.zeros([samples_4d.shape[0], C, target_latent_height, target_latent_width], 
                              device=device, dtype=torch.float32)
        F_embed[:, :, :H, :W] = F_low
        
        # Generate Taper mask
        mask = make_frequency_mask(H, W, target_latent_height, target_latent_width, taper, device)
        
        # Convert sigma to flow-matching interpolation factor t ∈ [0, 1] (Paper Eq 5 & 6)
        if edm_style and sigma > 0.0:
            t = sigma / (1.0 + sigma)
        else:
            t = sigma
            
        r = scale_factor
        # t_aligned is the effective noise level at the higher resolution
        t_aligned = (r * t) / (1.0 + (r - 1.0) * t)
        # kappa corrects for amplitude reduction from zero-padded DCT upsampling
        kappa = r / (1.0 + (r - 1.0) * t)
        
        # Convert t_aligned back to sigma_aligned for the next scheduler phase
        if edm_style:
            sigma_aligned = t_aligned / max(1.0 - t_aligned, 1e-6)
        else:
            sigma_aligned = t_aligned
        
        # Generate independent high-frequency noise
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn([samples_4d.shape[0], C, target_latent_height, target_latent_width], 
                            generator=gen, device=device, dtype=torch.float32)
        
        scaled_noise = sigma * noise_strength * noise
        
        # Blend low-frequency and high-frequency regions
        if blend_mode == "hard":
            F_high = mask * F_embed + (1.0 - mask) * scaled_noise
        elif blend_mode == "linear":
            F_high = mask * F_embed + (1.0 - mask) * scaled_noise
        elif blend_mode == "variance_preserving":
            # clamp to avoid numerical issues
            F_high = mask * F_embed + torch.sqrt(torch.clamp(1.0 - mask**2, min=0.0)) * scaled_noise
        else:
            F_high = mask * F_embed + (1.0 - mask) * scaled_noise
            
        # Inverse DCT back to spatial domain
        x_high = idct2(F_high)
        
        # Apply amplitude correction (kappa)
        x_high = kappa * x_high
        
        # Reshape back to 5D if needed
        if is_5d:
            # Reshape [B * T, C, H_new, W_new] -> [B, T, C, H_new, W_new] -> permute(0, 2, 1, 3, 4) -> [B, C, T, H_new, W_new]
            x_high_out = x_high.reshape(B, T, C, target_latent_height, target_latent_width).permute(0, 2, 1, 3, 4)
        else:
            x_high_out = x_high
        
        # Reconstruct ComfyUI latent dict
        out_latent = latent.copy()
        out_latent["samples"] = x_high_out.to(orig_dtype)
        
        return (out_latent, float(sigma_aligned))



class SplitSigmaArrayLSE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "split_step": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("high_sigmas", "low_sigmas", "high_steps", "low_steps", "transition_sigma")
    FUNCTION = "split"
    CATEGORY = "latent/spectral"

    def split(self, sigmas, split_step):
        total_len = len(sigmas)
        
        # Ensure split_step is within valid range
        # sigmas normally has length (steps + 1)
        actual_split = min(max(1, split_step), total_len - 2)
        
        high_sigmas = sigmas[:actual_split + 1]
        low_sigmas = sigmas[actual_split:]
        
        high_steps = actual_split
        low_steps = total_len - 1 - actual_split
        transition_sigma = float(sigmas[actual_split].item())
        
        return (high_sigmas, low_sigmas, high_steps, low_steps, transition_sigma)


class SplitSigmaArrayDenoiseLSE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("high_sigmas", "low_sigmas", "high_steps", "low_steps", "transition_sigma")
    FUNCTION = "split"
    CATEGORY = "latent/spectral"

    def split(self, sigmas, denoise):
        total_len = len(sigmas)
        steps = max(total_len - 1, 0)
        
        # Replicating native ComfyUI SplitSigmasDenoise math exactly
        total_steps = int(round(steps * denoise))
        
        # Ensure we have a valid split
        total_steps = min(max(1, total_steps), steps - 1)
        
        actual_split = steps - total_steps
        
        high_sigmas = sigmas[:actual_split + 1]
        low_sigmas = sigmas[actual_split:]
        
        high_steps = actual_split
        low_steps = total_steps
        transition_sigma = float(sigmas[actual_split].item())
        
        return (high_sigmas, low_sigmas, high_steps, low_steps, transition_sigma)




