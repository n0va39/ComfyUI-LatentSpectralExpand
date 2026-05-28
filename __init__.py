from .latent_spectral_expand import (
    LatentSpectralExpand,
    LSESegmentSigmaPlanner,
    LSEStagePrepare,
    LSESegmentSigmas2,
    LSESegmentSigmas3,
    SplitSigmaArrayLSE,
    SplitSigmaArrayDenoiseLSE,
)

NODE_CLASS_MAPPINGS = {
    "LatentSpectralExpand": LatentSpectralExpand,
    "LSESegmentSigmaPlanner": LSESegmentSigmaPlanner,
    "LSEStagePrepare": LSEStagePrepare,
    "LSESegmentSigmas2": LSESegmentSigmas2,
    "LSESegmentSigmas3": LSESegmentSigmas3,
    "SplitSigmaArrayLSE": SplitSigmaArrayLSE,
    "SplitSigmaArrayDenoiseLSE": SplitSigmaArrayDenoiseLSE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSpectralExpand": "Latent Spectral Expand (LSE)",
    "LSESegmentSigmaPlanner": "LSE Segment Sigma Planner",
    "LSEStagePrepare": "LSE Stage Prepare",
    "LSESegmentSigmas2": "LSE Segment Sigmas 2",
    "LSESegmentSigmas3": "LSE Segment Sigmas 3",
    "SplitSigmaArrayLSE": "Split Sigma Array (LSE)",
    "SplitSigmaArrayDenoiseLSE": "Split Sigma Array Denoise (LSE)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
