import torch
import sys
import os

# Add directory to sys.path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from latent_spectral_expand import LatentSpectralExpand, SplitSigmaArrayLSE, SplitSigmaArrayDenoiseLSE, dct2, idct2

def test():
    # 1. Test numerical correctness of dct2 / idct2
    torch.manual_seed(42)
    x = torch.randn(2, 4, 64, 64)
    X = dct2(x)
    x_rec = idct2(X)
    
    error = torch.abs(x - x_rec).max().item()
    print(f"DCT Reconstruction Error: {error}")
    assert error < 1e-4, "DCT reconstruction error is too high!"
    print("[PASS] DCT/IDCT is numerically stable and exact.")

    # 2. Test LatentSpectralExpand node
    node = LatentSpectralExpand()
    
    latent = {"samples": torch.randn(1, 4, 64, 64)}
    
    # Expand to 128x128 (scale_factor=2.0)
    out_latent_dict, sigma_aligned = node.expand(
        latent=latent,
        scale_factor=2.0,
        sigma=1.0,
        noise_strength=1.0,
        seed=123,
        taper=8,
        blend_mode="variance_preserving",
        edm_style=True
    )
    
    out_latent = out_latent_dict["samples"]
    print(f"Original shape: {latent['samples'].shape}")
    print(f"Expanded shape: {out_latent.shape}")
    print(f"Sigma aligned: {sigma_aligned}")
    
    assert out_latent.shape == (1, 4, 128, 128), "Output shape mismatch!"
    assert abs(sigma_aligned - 2.0) < 1e-4, "sigma_aligned calculation incorrect!"
    print("[PASS] Latent expansion shape and aligned sigma are correct.")
    
    # 3. Test SplitSigmaArrayLSE node
    split_node = SplitSigmaArrayLSE()
    sigmas = torch.linspace(10.0, 0.0, 11)
    
    high_sigmas, low_sigmas, high_steps, low_steps, transition_sigma = split_node.split(
        sigmas=sigmas,
        split_step=4
    )
    
    print(f"Original sigmas: {sigmas.tolist()}")
    print(f"High sigmas: {high_sigmas.tolist()} (steps: {high_steps})")
    print(f"Low sigmas: {low_sigmas.tolist()} (steps: {low_steps})")
    print(f"Transition sigma: {transition_sigma}")
    
    assert len(high_sigmas) == 5, "High sigmas length mismatch!"
    assert len(low_sigmas) == 7, "Low sigmas length mismatch!"
    assert high_steps == 4, "High steps mismatch!"
    assert low_steps == 6, "Low steps mismatch!"
    assert transition_sigma == float(sigmas[4].item()), "Transition sigma mismatch!"
    print("[PASS] SplitSigmaArrayLSE functions correctly.")

    # 4. Test SplitSigmaArrayDenoiseLSE node
    split_denoise_node = SplitSigmaArrayDenoiseLSE()
    
    high_sigmas_d, low_sigmas_d, high_steps_d, low_steps_d, transition_sigma_d = split_denoise_node.split(
        sigmas=sigmas,
        denoise=0.4
    )
    
    print(f"Denoise split (0.4) - High steps: {high_steps_d}, Low steps: {low_steps_d}, Transition sigma: {transition_sigma_d}")
    
    assert high_steps_d == 6, "Denoise high steps mismatch!"
    assert low_steps_d == 4, "Denoise low steps mismatch!"
    assert transition_sigma_d == float(sigmas[6].item()), "Denoise transition sigma mismatch!"
    print("[PASS] SplitSigmaArrayDenoiseLSE functions correctly.")

    # 5. Test 5D Video/Audio Latent Expansion
    latent_5d = {"samples": torch.randn(1, 4, 8, 64, 64)}
    out_latent_dict_5d, sigma_aligned_5d = node.expand(
        latent=latent_5d,
        scale_factor=2.0,
        sigma=1.0,
        noise_strength=1.0,
        seed=123,
        taper=8,
        blend_mode="variance_preserving",
        edm_style=True
    )
    out_latent_5d = out_latent_dict_5d["samples"]
    print(f"Original 5D shape: {latent_5d['samples'].shape}")
    print(f"Expanded 5D shape: {out_latent_5d.shape}")
    assert out_latent_5d.shape == (1, 4, 8, 128, 128), "5D Output shape mismatch!"
    print("[PASS] 5D Latent expansion shape is correct.")
    
if __name__ == "__main__":
    test()


