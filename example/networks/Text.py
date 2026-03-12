import json
from PIL import Image
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

model_name = "biomedclip_local"

with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]
if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)
model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained=None,
)
checkpoint_path = "checkpoints/open_clip_pytorch_model.bin"

checkpoint = torch.load(checkpoint_path, map_location='cpu')

model.load_state_dict(checkpoint, strict=True)