from .latent_spectral_expand import LatentSpectralExpand, SplitSigmaArrayLSE, SplitSigmaArrayDenoiseLSE

NODE_CLASS_MAPPINGS = {
    "LatentSpectralExpand": LatentSpectralExpand,
    "SplitSigmaArrayLSE": SplitSigmaArrayLSE,
    "SplitSigmaArrayDenoiseLSE": SplitSigmaArrayDenoiseLSE
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSpectralExpand": "Latent Spectral Expand (LSE)",
    "SplitSigmaArrayLSE": "Split Sigma Array (LSE)",
    "SplitSigmaArrayDenoiseLSE": "Split Sigma Array Denoise (LSE)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


