import torch
import math


LSE_CONTEXT_TYPE = "LSE_CONTEXT"
LATENT_SIZE_MULTIPLE = 2  # Anima / DiT latent H/W multiple.


def snap_to_multiple(value, multiple=LATENT_SIZE_MULTIPLE):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def get_dct_matrix(N, dtype=torch.float32, device='cpu'):
    """
    Generate an N x N orthonormal DCT-II matrix.
    """
    n = torch.arange(N, dtype=dtype, device=device)
    k = n.view(-1, 1)
    dct = torch.cos(math.pi / N * (n + 0.5) * k)
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
        taper_h = min(taper, orig_h)
        taper_w = min(taper, orig_w)
        mask_h[:orig_h - taper_h] = 1.0
        mask_w[:orig_w - taper_w] = 1.0

        for i in range(taper_h):
            pos = i / taper_h
            mask_h[orig_h - taper_h + i] = 0.5 * (1.0 + math.cos(math.pi * pos))

        for i in range(taper_w):
            pos = i / taper_w
            mask_w[orig_w - taper_w + i] = 0.5 * (1.0 + math.cos(math.pi * pos))

    mask = torch.outer(mask_h, mask_w)
    return mask.unsqueeze(0).unsqueeze(0)


# -----------------------------------------------------------------------------
# LSE/SPD scheduler utility functions
# -----------------------------------------------------------------------------

def sigma_to_t(sigma, edm_style=False):
    sigma = float(sigma)
    if edm_style:
        return sigma / (1.0 + sigma)
    return sigma


def t_to_sigma(t, edm_style=False):
    t = float(t)
    if edm_style:
        if t <= 0.0:
            return 0.0
        t = min(t, 1.0 - 1e-6)
        return t / max(1.0 - t, 1e-6)
    return t


def align_t(t, scale_ratio):
    """
    Paper-style timestep alignment used after spectral expansion.
    """
    t = float(t)
    r = float(scale_ratio)
    return (r * t) / (1.0 + (r - 1.0) * t)


def parse_float_list(value, name):
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError(f"{name} is empty.")
        return [float(p) for p in parts]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise ValueError(f"{name} must be a comma-separated string or list.")


def make_t_uniform_sigmas(start_t, end_t, steps, edm_style=False, device="cpu"):
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1 for a sigma segment.")
    start_t = float(start_t)
    end_t = float(end_t)
    if start_t <= end_t:
        raise ValueError(f"segment start_t must be greater than end_t, got {start_t} <= {end_t}.")

    ts = torch.linspace(start_t, end_t, steps + 1, dtype=torch.float32, device=device)
    if edm_style:
        ts = torch.clamp(ts, 0.0, 1.0 - 1e-6)
        sigmas = ts / torch.clamp(1.0 - ts, min=1e-6)
    else:
        sigmas = ts

    if end_t <= 1e-8:
        sigmas[-1] = 0.0
    else:
        sigmas[-1] = float(t_to_sigma(end_t, edm_style=edm_style))
    sigmas[0] = float(t_to_sigma(start_t, edm_style=edm_style))
    return sigmas


def _curve_position_for_t(base_t_values, t):
    """
    Return normalized position u in [0, 1] for a descending base t curve.
    u=0 is the first sigma and u=1 is the final sigma.
    """
    t = float(t)
    n = len(base_t_values) - 1
    if n <= 0:
        return 0.0
    if t >= base_t_values[0]:
        return 0.0
    if t <= base_t_values[-1]:
        return 1.0

    for i in range(n):
        hi = float(base_t_values[i])
        lo = float(base_t_values[i + 1])
        if hi >= t >= lo:
            denom = hi - lo
            frac = 0.0 if abs(denom) < 1e-12 else (hi - t) / denom
            return (i + frac) / n
    return 1.0


def _curve_t_at_position(base_t_values, u):
    n = len(base_t_values) - 1
    if n <= 0:
        return float(base_t_values[0])
    u = min(max(float(u), 0.0), 1.0)
    x = u * n
    i = int(math.floor(x))
    if i >= n:
        return float(base_t_values[-1])
    frac = x - i
    return float(base_t_values[i]) * (1.0 - frac) + float(base_t_values[i + 1]) * frac


def make_base_curve_sigmas(base_t_values, start_u, end_u, steps, edm_style=False, device="cpu"):
    """
    Generate a segment by re-sampling the original base sigma curve shape.
    This preserves the scheduler curvature instead of using a straight t-linear segment.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1 for a sigma segment.")
    if start_u >= end_u:
        raise ValueError(f"segment start_u must be smaller than end_u, got {start_u} >= {end_u}.")

    us = torch.linspace(float(start_u), float(end_u), steps + 1, dtype=torch.float32, device=device)
    ts = [_curve_t_at_position(base_t_values, float(u.item())) for u in us]
    sigmas = torch.tensor([t_to_sigma(t, edm_style=edm_style) for t in ts], dtype=torch.float32, device=device)
    if ts[-1] <= 1e-8:
        sigmas[-1] = 0.0
    return sigmas


def allocate_steps_fixed_total(lengths, total_steps):
    n = len(lengths)
    total_steps = int(total_steps)
    if total_steps < n:
        raise ValueError(f"total_steps ({total_steps}) must be >= segment count ({n}).")

    total_length = sum(lengths)
    if total_length <= 0:
        raise ValueError("Total segment length must be positive.")

    raw = [total_steps * (l / total_length) for l in lengths]
    steps = [max(1, int(math.floor(v))) for v in raw]

    diff = total_steps - sum(steps)
    frac_order_desc = sorted(range(n), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    frac_order_asc = list(reversed(frac_order_desc))

    while diff > 0:
        for i in frac_order_desc:
            steps[i] += 1
            diff -= 1
            if diff == 0:
                break

    while diff < 0:
        changed = False
        for i in frac_order_asc:
            if steps[i] > 1:
                steps[i] -= 1
                diff += 1
                changed = True
                if diff == 0:
                    break
        if not changed:
            raise ValueError("Could not reduce segment steps while keeping each segment >= 1.")

    return steps


def allocate_steps_preserve_dt(lengths, initial_t, final_t, base_steps):
    base_steps = int(base_steps)
    if base_steps < 1:
        raise ValueError("base_steps must be at least 1.")
    base_dt = (float(initial_t) - float(final_t)) / base_steps
    if base_dt <= 0:
        raise ValueError("base_dt must be positive.")
    return [max(1, int(math.ceil(l / base_dt))) for l in lengths]


def allocate_steps_preserve_curve_du(lengths_u, base_steps):
    base_steps = int(base_steps)
    if base_steps < 1:
        raise ValueError("base_steps must be at least 1.")
    base_du = 1.0 / base_steps
    return [max(1, int(math.ceil(l / base_du))) for l in lengths_u]


def build_lse_context(
    base_sigmas,
    scale_schedule,
    transition_list,
    transition_mode,
    step_policy,
    scheduler_mode,
    noise_strength,
    taper,
    blend_mode,
    edm_style,
    seed_mode,
    seed,
):
    if scheduler_mode not in ("base_curve", "t_uniform"):
        raise ValueError("scheduler_mode must be 'base_curve' or 't_uniform'.")

    scales = parse_float_list(scale_schedule, "scale_schedule")
    transitions_raw = parse_float_list(transition_list, "transition_list")

    if len(scales) < 1:
        raise ValueError("scale_schedule must contain at least one scale.")
    if len(transitions_raw) != max(0, len(scales) - 1):
        raise ValueError("transition_list length must equal len(scale_schedule) - 1.")
    if any(s <= 0 for s in scales):
        raise ValueError("All scale values must be positive.")
    for i in range(1, len(scales)):
        if scales[i] < scales[i - 1]:
            raise ValueError("scale_schedule must be non-decreasing.")

    device = base_sigmas.device if hasattr(base_sigmas, "device") else "cpu"
    base_sigmas_cpu = base_sigmas.detach().cpu() if hasattr(base_sigmas, "detach") else base_sigmas
    base_total_steps = max(int(len(base_sigmas_cpu) - 1), 1)
    base_sigma_values = [float(v.item() if hasattr(v, "item") else v) for v in base_sigmas_cpu]
    base_t_values = [sigma_to_t(v, edm_style=edm_style) for v in base_sigma_values]

    initial_sigma = float(base_sigma_values[0])
    final_sigma = float(base_sigma_values[-1])
    initial_t = float(base_t_values[0])
    final_t = float(base_t_values[-1])
    if final_t < 1e-8:
        final_t = 0.0
        base_t_values[-1] = 0.0

    if transition_mode == "sigma":
        transitions_t = [sigma_to_t(v, edm_style=edm_style) for v in transitions_raw]
        transitions_sigma = [float(v) for v in transitions_raw]
    elif transition_mode == "t":
        transitions_t = [float(v) for v in transitions_raw]
        transitions_sigma = [t_to_sigma(v, edm_style=edm_style) for v in transitions_t]
    else:
        raise ValueError("transition_mode must be 't' or 'sigma'.")

    previous_t = initial_t
    for t in transitions_t:
        if not (previous_t > t > final_t - 1e-8):
            raise ValueError(
                f"Transitions must be strictly decreasing between initial_t and final_t. "
                f"Got initial_t={initial_t:.6g}, final_t={final_t:.6g}, transitions={transitions_t}."
            )
        previous_t = t

    segments_meta = []
    for i, scale in enumerate(scales):
        if i == 0:
            start_t = initial_t
        else:
            ratio = scales[i] / scales[i - 1]
            start_t = align_t(transitions_t[i - 1], ratio)

        end_t = transitions_t[i] if i < len(transitions_t) else final_t
        if start_t <= end_t:
            raise ValueError(
                f"Invalid stage {i}: aligned start_t ({start_t}) must be greater than end_t ({end_t}). "
                "Try moving transitions later or reducing scale jumps."
            )

        start_u = _curve_position_for_t(base_t_values, start_t)
        end_u = _curve_position_for_t(base_t_values, end_t)
        if scheduler_mode == "base_curve" and start_u >= end_u:
            raise ValueError(
                f"Invalid stage {i}: base-curve start_u ({start_u}) must be smaller than end_u ({end_u})."
            )

        segments_meta.append({
            "stage_index": i,
            "start_t": float(start_t),
            "end_t": float(end_t),
            "start_u": float(start_u),
            "end_u": float(end_u),
            "scale": float(scale),
            "is_last": i == len(scales) - 1,
        })

    if scheduler_mode == "base_curve":
        lengths = [seg["end_u"] - seg["start_u"] for seg in segments_meta]
        if step_policy == "fixed_total_steps":
            segment_steps = allocate_steps_fixed_total(lengths, base_total_steps)
        elif step_policy == "preserve_dt":
            segment_steps = allocate_steps_preserve_curve_du(lengths, base_total_steps)
        else:
            raise ValueError("step_policy must be 'fixed_total_steps' or 'preserve_dt'.")
    else:
        lengths = [seg["start_t"] - seg["end_t"] for seg in segments_meta]
        if step_policy == "fixed_total_steps":
            segment_steps = allocate_steps_fixed_total(lengths, base_total_steps)
        elif step_policy == "preserve_dt":
            segment_steps = allocate_steps_preserve_dt(lengths, initial_t, final_t, base_total_steps)
        else:
            raise ValueError("step_policy must be 'fixed_total_steps' or 'preserve_dt'.")

    for seg, steps in zip(segments_meta, segment_steps):
        if scheduler_mode == "base_curve":
            sigmas = make_base_curve_sigmas(
                base_t_values,
                seg["start_u"],
                seg["end_u"],
                steps,
                edm_style=edm_style,
                device=device,
            )
        else:
            sigmas = make_t_uniform_sigmas(seg["start_t"], seg["end_t"], steps, edm_style=edm_style, device=device)
        seg["sigmas"] = sigmas
        seg["steps"] = int(steps)
        seg["start_sigma"] = float(sigmas[0].item())
        seg["end_sigma"] = float(sigmas[-1].item())

    context = {
        "version": 3,
        "latent_size_multiple": LATENT_SIZE_MULTIPLE,
        "stage0_init": "dct_lowpass",
        "step_policy": step_policy,
        "scheduler_mode": scheduler_mode,
        "transition_mode": transition_mode,
        "base_total_steps": int(base_total_steps),
        "actual_total_steps": int(sum(segment_steps)),
        "initial_sigma": initial_sigma,
        "final_sigma": final_sigma,
        "initial_t": float(initial_t),
        "final_t": float(final_t),
        "scales": scales,
        "transition_t": [float(v) for v in transitions_t],
        "transition_sigma": [float(v) for v in transitions_sigma],
        "base_curve_t": [float(v) for v in base_t_values],
        "expand_settings": {
            "noise_strength": float(noise_strength),
            "taper": int(taper),
            "blend_mode": blend_mode,
            "edm_style": bool(edm_style),
            "seed_mode": seed_mode,
            "seed": int(seed),
        },
        "segments": segments_meta,
    }
    return context


def get_stage_seed(base_seed, seed_mode, stage_index):
    base_seed = int(base_seed)
    if seed_mode == "fixed":
        return base_seed
    if seed_mode == "per_stage_offset":
        return (base_seed + int(stage_index)) & 0xffffffffffffffff
    if seed_mode == "random":
        return int(torch.seed()) & 0xffffffffffffffff
    return (base_seed + int(stage_index)) & 0xffffffffffffffff


def _latent_to_4d(samples):
    is_5d = (samples.ndim == 5)
    if is_5d:
        B, C, T, H, W = samples.shape
        samples_4d = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        shape_info = (is_5d, B, C, T, H, W)
    else:
        B, C, H, W = samples.shape
        samples_4d = samples
        shape_info = (is_5d, B, C, None, H, W)
    return samples_4d, shape_info


def _latent_from_4d(samples_4d, shape_info, target_h, target_w):
    is_5d, B, C, T, _, _ = shape_info
    if is_5d:
        return samples_4d.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)
    return samples_4d


def dct_lowpass_latent(latent, scale):
    """
    Initialize stage 0 by low-pass filtering a full-size latent to the requested
    stage scale. This matches the SPD prefix idea: users provide the final-size
    latent, and stage 0 starts on a smaller DCT low-pass grid.
    """
    scale = float(scale)
    samples = latent["samples"]
    orig_dtype = samples.dtype
    samples_4d, shape_info = _latent_to_4d(samples)
    _, _, _, _, H, W = shape_info

    target_h = min(snap_to_multiple(H * scale), H)
    target_w = min(snap_to_multiple(W * scale), W)

    if target_h >= H and target_w >= W:
        return latent, 1.0

    x32 = samples_4d.to(torch.float32)
    F_full = dct2(x32)
    x_low = idct2(F_full[:, :, :target_h, :target_w])
    x_low_out = _latent_from_4d(x_low, shape_info, target_h, target_w)

    out_latent = latent.copy()
    out_latent["samples"] = x_low_out.to(orig_dtype)
    return out_latent, math.sqrt((target_h / H) * (target_w / W))


def spectral_expand_latent(
    latent,
    scale_factor,
    sigma,
    noise_strength,
    seed,
    taper,
    blend_mode,
    edm_style,
    use_actual_scale_for_alignment=False,
):
    samples = latent["samples"]
    orig_dtype = samples.dtype
    device = samples.device
    samples_4d, shape_info = _latent_to_4d(samples)
    _, _, C, _, H, W = shape_info

    target_latent_height = max(H, snap_to_multiple(H * float(scale_factor)))
    target_latent_width = max(W, snap_to_multiple(W * float(scale_factor)))

    if target_latent_height <= H and target_latent_width <= W:
        return latent, float(sigma), 1.0

    x32 = samples_4d.to(torch.float32)
    F_low = dct2(x32)

    F_embed = torch.zeros(
        [samples_4d.shape[0], C, target_latent_height, target_latent_width],
        device=device,
        dtype=torch.float32,
    )
    F_embed[:, :, :H, :W] = F_low

    mask = make_frequency_mask(H, W, target_latent_height, target_latent_width, taper, device)

    r_h = target_latent_height / H
    r_w = target_latent_width / W
    r_eff = math.sqrt(r_h * r_w)
    r = r_eff if use_actual_scale_for_alignment else float(scale_factor)

    if edm_style and sigma > 0.0:
        t = sigma / (1.0 + sigma)
    else:
        t = sigma

    t_aligned = (r * t) / (1.0 + (r - 1.0) * t)
    kappa = r / (1.0 + (r - 1.0) * t)

    if edm_style:
        sigma_aligned = t_aligned / max(1.0 - t_aligned, 1e-6)
    else:
        sigma_aligned = t_aligned

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    noise = torch.randn(
        [samples_4d.shape[0], C, target_latent_height, target_latent_width],
        generator=gen,
        device=device,
        dtype=torch.float32,
    )

    scaled_noise = float(sigma) * float(noise_strength) * noise

    if blend_mode == "hard":
        F_high = mask * F_embed + (1.0 - mask) * scaled_noise
    elif blend_mode == "linear":
        F_high = mask * F_embed + (1.0 - mask) * scaled_noise
    elif blend_mode == "variance_preserving":
        F_high = mask * F_embed + torch.sqrt(torch.clamp(1.0 - mask**2, min=0.0)) * scaled_noise
    else:
        F_high = mask * F_embed + (1.0 - mask) * scaled_noise

    x_high = idct2(F_high)
    x_high = kappa * x_high

    x_high_out = _latent_from_4d(x_high, shape_info, target_latent_height, target_latent_width)
    out_latent = latent.copy()
    out_latent["samples"] = x_high_out.to(orig_dtype)
    return out_latent, float(sigma_aligned), float(r_eff)


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
                "edm_style": ("BOOLEAN", {"default": False, "tooltip": "False is recommended for Anima/DiT sigma schedules in [0, 1]. True converts EDM-style sigma to t=sigma/(1+sigma)."})
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "sigma_aligned")
    FUNCTION = "expand"
    CATEGORY = "latent/spectral"

    def expand(self, latent, scale_factor, sigma, noise_strength, seed, taper, blend_mode, edm_style):
        out_latent, sigma_aligned, _ = spectral_expand_latent(
            latent=latent,
            scale_factor=scale_factor,
            sigma=sigma,
            noise_strength=noise_strength,
            seed=seed,
            taper=taper,
            blend_mode=blend_mode,
            edm_style=edm_style,
            use_actual_scale_for_alignment=False,
        )
        return (out_latent, sigma_aligned)


class LSESegmentSigmaPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sigmas": ("SIGMAS", ),
                "scale_schedule": ("STRING", {"default": "0.5,0.75,1.0", "multiline": False}),
                "transition_list": ("STRING", {"default": "0.7,0.4", "multiline": False}),
                "transition_mode": (["sigma", "t"], {"default": "sigma"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["base_curve", "t_uniform"], {"default": "base_curve"}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "taper": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "blend_mode": (["variance_preserving", "linear", "hard"], {"default": "variance_preserving"}),
                "edm_style": ("BOOLEAN", {"default": False}),
                "seed_mode": (["fixed", "per_stage_offset", "random"], {"default": "per_stage_offset"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (LSE_CONTEXT_TYPE, "INT", "INT")
    RETURN_NAMES = ("lse_context", "actual_total_steps", "segment_count")
    FUNCTION = "plan"
    CATEGORY = "latent/spectral"

    def plan(
        self,
        base_sigmas,
        scale_schedule,
        transition_list,
        transition_mode,
        step_policy,
        scheduler_mode,
        noise_strength,
        taper,
        blend_mode,
        edm_style,
        seed_mode,
        seed,
    ):
        context = build_lse_context(
            base_sigmas=base_sigmas,
            scale_schedule=scale_schedule,
            transition_list=transition_list,
            transition_mode=transition_mode,
            step_policy=step_policy,
            scheduler_mode=scheduler_mode,
            noise_strength=noise_strength,
            taper=taper,
            blend_mode=blend_mode,
            edm_style=edm_style,
            seed_mode=seed_mode,
            seed=seed,
        )
        return (context, int(context["actual_total_steps"]), int(len(context["segments"])))


class LSEStagePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lse_context": (LSE_CONTEXT_TYPE, ),
                "latent": ("LATENT", ),
                "stage_index": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "INT", "FLOAT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("processed_latent", "stage_sigmas", "stage_steps", "current_scale", "transition_sigma_used", "is_last_stage")
    FUNCTION = "prepare"
    CATEGORY = "latent/spectral"

    def prepare(self, lse_context, latent, stage_index):
        segments = lse_context["segments"]
        stage_index = int(stage_index)
        if stage_index < 0 or stage_index >= len(segments):
            raise ValueError(f"stage_index {stage_index} is out of range for {len(segments)} segments.")

        segment = segments[stage_index]
        processed_latent = latent
        transition_sigma_used = 0.0

        if stage_index == 0:
            scale = float(segment["scale"])
            if scale < 1.0:
                processed_latent, _ = dct_lowpass_latent(latent, scale)
        else:
            prev_segment = segments[stage_index - 1]
            prev_scale = float(prev_segment["scale"])
            curr_scale = float(segment["scale"])
            if prev_scale <= 0:
                raise ValueError("Previous scale must be positive.")
            scale_factor = curr_scale / prev_scale
            transition_sigma_used = float(lse_context["transition_sigma"][stage_index - 1])
            settings = lse_context["expand_settings"]
            expand_seed = get_stage_seed(settings["seed"], settings["seed_mode"], stage_index)

            processed_latent, _, _ = spectral_expand_latent(
                latent=latent,
                scale_factor=scale_factor,
                sigma=transition_sigma_used,
                noise_strength=settings["noise_strength"],
                seed=expand_seed,
                taper=settings["taper"],
                blend_mode=settings["blend_mode"],
                edm_style=settings["edm_style"],
                use_actual_scale_for_alignment=True,
            )

        return (
            processed_latent,
            segment["sigmas"],
            int(segment["steps"]),
            float(segment["scale"]),
            float(transition_sigma_used),
            bool(segment["is_last"]),
        )


class LSESegmentSigmas2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sigmas": ("SIGMAS", ),
                "scale_0": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 10.0, "step": 0.05}),
                "scale_1": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05}),
                "transition": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_mode": (["sigma", "t"], {"default": "sigma"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["base_curve", "t_uniform"], {"default": "base_curve"}),
                "edm_style": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "FLOAT", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("seg0_sigmas", "seg1_sigmas", "transition_sigma_0", "next_scale_factor_0", "seg0_steps", "seg1_steps", "actual_total_steps")
    FUNCTION = "make"
    CATEGORY = "latent/spectral"

    def make(self, base_sigmas, scale_0, scale_1, transition, transition_mode, step_policy, scheduler_mode, edm_style):
        context = build_lse_context(
            base_sigmas=base_sigmas,
            scale_schedule=f"{scale_0},{scale_1}",
            transition_list=str(transition),
            transition_mode=transition_mode,
            step_policy=step_policy,
            scheduler_mode=scheduler_mode,
            noise_strength=1.0,
            taper=8,
            blend_mode="variance_preserving",
            edm_style=edm_style,
            seed_mode="per_stage_offset",
            seed=0,
        )
        seg0, seg1 = context["segments"][0], context["segments"][1]
        return (
            seg0["sigmas"],
            seg1["sigmas"],
            float(context["transition_sigma"][0]),
            float(scale_1 / scale_0),
            int(seg0["steps"]),
            int(seg1["steps"]),
            int(context["actual_total_steps"]),
        )


class LSESegmentSigmas3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sigmas": ("SIGMAS", ),
                "scale_0": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 10.0, "step": 0.05}),
                "scale_1": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 10.0, "step": 0.05}),
                "scale_2": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05}),
                "transition_0": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_1": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_mode": (["sigma", "t"], {"default": "sigma"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["base_curve", "t_uniform"], {"default": "base_curve"}),
                "edm_style": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "SIGMAS", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "seg0_sigmas", "seg1_sigmas", "seg2_sigmas",
        "transition_sigma_0", "transition_sigma_1",
        "next_scale_factor_0", "next_scale_factor_1",
        "seg0_steps", "seg1_steps", "seg2_steps", "actual_total_steps"
    )
    FUNCTION = "make"
    CATEGORY = "latent/spectral"

    def make(self, base_sigmas, scale_0, scale_1, scale_2, transition_0, transition_1, transition_mode, step_policy, scheduler_mode, edm_style):
        context = build_lse_context(
            base_sigmas=base_sigmas,
            scale_schedule=f"{scale_0},{scale_1},{scale_2}",
            transition_list=f"{transition_0},{transition_1}",
            transition_mode=transition_mode,
            step_policy=step_policy,
            scheduler_mode=scheduler_mode,
            noise_strength=1.0,
            taper=8,
            blend_mode="variance_preserving",
            edm_style=edm_style,
            seed_mode="per_stage_offset",
            seed=0,
        )
        seg0, seg1, seg2 = context["segments"][0], context["segments"][1], context["segments"][2]
        return (
            seg0["sigmas"],
            seg1["sigmas"],
            seg2["sigmas"],
            float(context["transition_sigma"][0]),
            float(context["transition_sigma"][1]),
            float(scale_1 / scale_0),
            float(scale_2 / scale_1),
            int(seg0["steps"]),
            int(seg1["steps"]),
            int(seg2["steps"]),
            int(context["actual_total_steps"]),
        )


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
        total_steps = int(round(steps * denoise))
        total_steps = min(max(1, total_steps), steps - 1)
        actual_split = steps - total_steps
        high_sigmas = sigmas[:actual_split + 1]
        low_sigmas = sigmas[actual_split:]
        high_steps = actual_split
        low_steps = total_steps
        transition_sigma = float(sigmas[actual_split].item())
        return (high_sigmas, low_sigmas, high_steps, low_steps, transition_sigma)
