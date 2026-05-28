import torch
import math


LSE_CONTEXT_TYPE = "LSE_CONTEXT"


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

def sigma_to_t(sigma, edm_style=True):
    sigma = float(sigma)
    if edm_style:
        return sigma / (1.0 + sigma)
    return sigma


def t_to_sigma(t, edm_style=True):
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


def make_t_uniform_sigmas(start_t, end_t, steps, edm_style=True, device="cpu"):
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
    if scheduler_mode != "t_uniform":
        raise ValueError("Only scheduler_mode='t_uniform' is currently supported.")

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

    initial_sigma = float(base_sigmas_cpu[0].item() if hasattr(base_sigmas_cpu[0], "item") else base_sigmas_cpu[0])
    final_sigma = float(base_sigmas_cpu[-1].item() if hasattr(base_sigmas_cpu[-1], "item") else base_sigmas_cpu[-1])
    initial_t = sigma_to_t(initial_sigma, edm_style=edm_style)
    final_t = sigma_to_t(final_sigma, edm_style=edm_style)
    if final_t < 1e-8:
        final_t = 0.0

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
            raise ValueError("Transitions must be strictly decreasing between initial_t and final_t.")
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

        segments_meta.append({
            "stage_index": i,
            "start_t": float(start_t),
            "end_t": float(end_t),
            "scale": float(scale),
            "is_last": i == len(scales) - 1,
        })

    lengths = [seg["start_t"] - seg["end_t"] for seg in segments_meta]
    if step_policy == "fixed_total_steps":
        segment_steps = allocate_steps_fixed_total(lengths, base_total_steps)
    elif step_policy == "preserve_dt":
        segment_steps = allocate_steps_preserve_dt(lengths, initial_t, final_t, base_total_steps)
    else:
        raise ValueError("step_policy must be 'fixed_total_steps' or 'preserve_dt'.")

    for seg, steps in zip(segments_meta, segment_steps):
        sigmas = make_t_uniform_sigmas(seg["start_t"], seg["end_t"], steps, edm_style=edm_style, device=device)
        seg["sigmas"] = sigmas
        seg["steps"] = int(steps)
        seg["start_sigma"] = float(sigmas[0].item())
        seg["end_sigma"] = float(sigmas[-1].item())

    context = {
        "version": 1,
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

    is_5d = (samples.ndim == 5)
    if is_5d:
        B, C, T, H, W = samples.shape
        samples_4d = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = samples.shape
        T = None
        samples_4d = samples

    target_latent_height = max(H, int(round((H * float(scale_factor)) / 8.0) * 8))
    target_latent_width = max(W, int(round((W * float(scale_factor)) / 8.0) * 8))

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

    if is_5d:
        x_high_out = x_high.reshape(B, T, C, target_latent_height, target_latent_width).permute(0, 2, 1, 3, 4)
    else:
        x_high_out = x_high

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
                "edm_style": ("BOOLEAN", {"default": True, "tooltip": "Convert sigma to t ∈ [0, 1] and back (recommended for EDM/Karras-style sigma schedules)"})
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
                "transition_list": ("STRING", {"default": "0.55,0.22", "multiline": False}),
                "transition_mode": (["t", "sigma"], {"default": "t"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["t_uniform"], {"default": "t_uniform"}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "taper": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "blend_mode": (["variance_preserving", "linear", "hard"], {"default": "variance_preserving"}),
                "edm_style": ("BOOLEAN", {"default": True}),
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

        if stage_index > 0:
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
                "transition": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_mode": (["t", "sigma"], {"default": "t"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["t_uniform"], {"default": "t_uniform"}),
                "edm_style": ("BOOLEAN", {"default": True}),
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
                "transition_0": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_1": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "transition_mode": (["t", "sigma"], {"default": "t"}),
                "step_policy": (["fixed_total_steps", "preserve_dt"], {"default": "fixed_total_steps"}),
                "scheduler_mode": (["t_uniform"], {"default": "t_uniform"}),
                "edm_style": ("BOOLEAN", {"default": True}),
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
