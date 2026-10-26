"""Deepfake image detector — CPU inference.

PRIMARY MODEL  (used by the Streamlit app)
------------------------------------------
Architecture : EfficientNetV2-S backbone (timm tf_efficientnetv2_s) +
               custom 4-layer MLP classifier head.
Checkpoint   : models/deepfake_efficientnetv2s.bin
               Downloaded from Arko007/deepfake-image-detector (HuggingFace,
               Apache-2.0 licence). Epoch-6 AUC reported: 0.9986.
Input size   : 380 × 380 RGB
Output       : single sigmoid logit → P(FAKE). Threshold 0.5.
Notes        : Sigmoid output is a model score, not a calibrated probability.

FALLBACK MODEL  (kept for reference, not loaded by default)
-----------------------------------------------------------
Architecture : ResNet18 (video.pth)
Issue        : The video.pth checkpoint was trained on FaceForensics++ video
               frames. Its score distribution is compressed into 0.30–0.43 on
               all inputs (std ≈ 0.003), so it never crosses the 0.5 threshold
               and always predicts REAL. It is retained but not used.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Primary model (EfficientNetV2-S) ────────────────────────────────────────
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_efficientnetv2s.bin"
DEFAULT_INPUT_SIZE = (380, 380)
DEEPFAKE_THRESHOLD = 0.5

# ── Fallback model (ResNet18 — kept, not used by default) ───────────────────
RESNET_MODEL_PATH  = PROJECT_ROOT / "models" / "video.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DetectorError(Exception):
    """Raised when an image cannot be read or the model cannot be used."""


def _device() -> torch.device:
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# EfficientNetV2-S architecture
# (must match the training code that produced deepfake_efficientnetv2s.bin)
# --------------------------------------------------------------------------- #
class _DeepfakeEfficientNet(nn.Module):
    """EfficientNetV2-S backbone + 4-layer MLP head → single sigmoid logit."""

    def __init__(self) -> None:
        super().__init__()
        # timm backbone with no built-in head (num_classes=0 → raw features)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=False, num_classes=0
        )
        # 4-layer MLP: matches the classifier structure in the HF checkpoint
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # → (B, 1) logit
        return self.classifier(self.backbone(x))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_image_model(model_path: str | Path | None = None) -> nn.Module:
    """Load the EfficientNetV2-S deepfake detector for CPU inference.

    Parameters
    ----------
    model_path:
        Path to the ``.bin`` checkpoint.  Defaults to
        ``models/deepfake_efficientnetv2s.bin``.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise DetectorError(
            f"Model file not found at {path.resolve()}.\n"
            "Expected models/deepfake_efficientnetv2s.bin\n"
            "Download it from:\n"
            "  https://huggingface.co/Arko007/deepfake-image-detector/resolve/main/pytorch_model.bin\n"
            "and save as models/deepfake_efficientnetv2s.bin"
        )

    device = _device()
    model = _DeepfakeEfficientNet().to(device)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        raise DetectorError(f"Could not read the model file: {exc}") from exc

    # The checkpoint is a plain state_dict (no wrapper keys).
    if not isinstance(checkpoint, dict):
        raise DetectorError("Unexpected checkpoint format (expected state_dict).")

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        raise DetectorError(
            f"Checkpoint is missing {len(missing)} keys (first: {missing[0]}). "
            "Architecture mismatch."
        )
    if unexpected:
        raise DetectorError(
            f"Checkpoint has {len(unexpected)} unexpected keys "
            f"(first: {unexpected[0]}). Architecture mismatch."
        )

    model.eval()
    return model


def _image_transform() -> transforms.Compose:
    h, w = DEFAULT_INPUT_SIZE
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def read_rgb_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk and return it as an RGB uint8 array."""
    path = Path(image_path)
    if not path.exists():
        raise DetectorError(f"Image file not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise DetectorError("Could not read the image. Use a valid JPG or PNG file.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_rgb_image(rgb_image: np.ndarray, model: nn.Module) -> dict:
    """Run inference and return label, score, and processing metadata.

    Returns
    -------
    dict with keys:
        label            : "REAL" | "DEEPFAKE"
        deepfake_score   : float in [0, 1]  (sigmoid of the single logit)
        class_scores     : {"REAL": float, "DEEPFAKE": float}
        input_size       : (H, W)
        color_space      : "RGB"
        preprocessing    : description string
        architecture     : description string
        device           : torch device string
        score_note       : disclaimer string
        rgb_image        : original rgb_image array (passed through)
    """
    if rgb_image is None or not isinstance(rgb_image, np.ndarray):
        raise DetectorError("No image data was provided.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise DetectorError("The detector expects a 3-channel RGB image.")

    device = next(model.parameters()).device
    tensor = _image_transform()(rgb_image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            logit = model(tensor)            # shape (1, 1)
            deepfake_score = float(torch.sigmoid(logit).item())
    except Exception as exc:
        raise DetectorError(f"Model inference failed: {exc}") from exc

    label = "DEEPFAKE" if deepfake_score >= DEEPFAKE_THRESHOLD else "REAL"
    h, w = DEFAULT_INPUT_SIZE
    return {
        "label": label,
        "deepfake_score": deepfake_score,
        "class_scores": {
            "REAL": round(1.0 - deepfake_score, 6),
            "DEEPFAKE": round(deepfake_score, 6),
        },
        "input_size": (h, w),
        "color_space": "RGB",
        "preprocessing": f"Resize {h}x{w}, ImageNet mean/std normalization",
        "architecture": "EfficientNetV2-S + MLP head (timm backbone, sigmoid output)",
        "device": str(device),
        "score_note": (
            "Sigmoid of the model's output logit. Reported AUC 0.9986 at epoch 6 "
            "on the training dataset. This is NOT a calibrated probability and may "
            "not generalise to all image types."
        ),
        "rgb_image": rgb_image,
    }


def predict_image_file(image_path: str | Path, model: nn.Module) -> dict:
    """Read an image from disk and return detection results."""
    rgb_image = read_rgb_image(image_path)
    return predict_rgb_image(rgb_image, model)
