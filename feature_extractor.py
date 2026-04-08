import requests
import os
from typing import Literal

import torch
from timm.models import create_model
from timm.data import resolve_model_data_config, create_transform


class UpperGIFeatureExtractor(torch.nn.Module):
    """Feature extractor for upper-GI endoscopy images using ViT/16 backbones.

    The module loads a DINOv3 model pretrained on the UpperGI-400K dataset from
    local weights (downloading the checkpoint if missing) and returns the last
    feature map. Optional global average pooling can be enabled to return
    flattened embeddings.

    Args:
        size: Backbone size variant ("small", "base", or "large").
        model_dir_path: Directory containing pretrained checkpoint files. If None,
            defaults to ``dinov3/pretrained`` relative to this file.
        use_global_pool: If True, applies global average pooling and returns features of shape ``[B, D]``;
            otherwise, returns the last feature map with shape ``[B, D, 16, 16]``.
        use_native_transform: If True, applies timm-provided inference transforms
            before forwarding through the vision model.
    """
    VISION_MODEL_DICT = {
        "small": "vit_small_patch16_dinov3_qkvb.lvd1689m",
        "base": "vit_base_patch16_dinov3_qkvb.lvd1689m",
        "large": "vit_large_patch16_dinov3_qkvb.lvd1689m",
    }

    def __init__(
        self,
        size: Literal["small", "base", "large"],
        model_dir_path: str | None = None,
        use_global_pool = True,
        use_native_transform = True
    ) -> None:
        super().__init__()

        self.vision_model_size = size
        self.use_native_transform = use_native_transform
        self.vision_model_name = UpperGIFeatureExtractor.VISION_MODEL_DICT[size]

        if model_dir_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir_path = os.path.join(module_dir, "dinov3", "pretrained")

        self.model_dir_path = model_dir_path
        self.model_file_name = f"dinov3-vit{self.vision_model_size[0]}16-pretrain-upperGI400k.pth"
        self.model_path = os.path.join(self.model_dir_path, self.model_file_name)

        if not os.path.exists(self.model_path):
            print(f"{self.__class__.__name__}: model file not found at {self.model_path}. Downloading now.")
            self._download_model()

        self.vision_model = create_model(
            self.vision_model_name,
            pretrained=True,
            features_only=True,
            pretrained_cfg_overlay={ "file": self.model_path }
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if use_global_pool else None
        
        # get model specific transforms (normalization, resize)
        data_config = resolve_model_data_config(self.vision_model)
        raw_transform = create_transform(**data_config, is_training=False)
        if isinstance(raw_transform, tuple):
            self.transform = raw_transform
        else:
            self.transform = (raw_transform,)

    def forward(self, x) -> torch.Tensor:
        if self.use_native_transform:
            for transform in self.transform:
                x = transform(x)

        output = self.vision_model(x)[-1]

        if self.global_pool is not None:
            output = self.global_pool(output)
            output = torch.flatten(output, 1)

        return output

    def _download_model(self) -> None:
        link = f"https://huggingface.co/tofriede/dinov3-upperGI/resolve/main/{self.model_file_name}"
        os.makedirs(self.model_dir_path, exist_ok=True)
        temp_model_path = f"{self.model_path}.tmp"

        try:
            with requests.get(link, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()

                with open(temp_model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # atomically move into place so interrupted downloads do not leave partial files.
            os.replace(temp_model_path, self.model_path)

        except requests.RequestException as exc:
            self._cleanup_temp_file(temp_model_path)
            raise RuntimeError(f"Failed to download model from {link}: {exc}") from exc
        except OSError as exc:
            self._cleanup_temp_file(temp_model_path)
            raise RuntimeError(f"Failed to write model file to {self.model_path}: {exc}") from exc

        print(f"{self.__class__.__name__}: model of size {self.vision_model_size} downloaded successfully.")

    @staticmethod
    def _cleanup_temp_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)