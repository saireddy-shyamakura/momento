import torch
import clip
from PIL import Image
import numpy as np
from config import DEVICE, MODEL_NAME

_model = None
_preprocess = None

def get_model():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return _model, _preprocess


def extract_image_features(image_path):
    model, preprocess = get_model()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {e}")

    with torch.inference_mode():
        image = preprocess(image).unsqueeze(0).to(DEVICE)

        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().numpy().astype(np.float32)


def extract_text_features(text):
    model, _ = get_model()

    with torch.inference_mode():
        tokens = clip.tokenize([text]).to(DEVICE)

        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().numpy().astype(np.float32)