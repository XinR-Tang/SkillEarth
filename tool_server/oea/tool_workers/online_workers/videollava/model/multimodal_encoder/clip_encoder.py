import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
)

import os
import json
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


# class CLIPVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()

#         self.is_loaded = False
#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = self._build_cfg_only()

#     def _read_raw_config(self):
#         cfg_path = os.path.join(self.vision_tower_name, "config.json")
#         if os.path.exists(cfg_path):
#             with open(cfg_path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         return {}

#     def _is_languagebind_image(self):
#         raw_cfg = self._read_raw_config()
#         model_type = raw_cfg.get("model_type", "")
#         return (
#             "LanguageBind_Image" in self.vision_tower_name
#             or model_type == "LanguageBindImage"
#         )

#     def _build_cfg_only(self):
#         # delay_load=True 时这里只是给外部读取 hidden_size / image_size / patch_size
#         # 不能再走 AutoConfig，否则会再次 KeyError: LanguageBindImage
#         if self._is_languagebind_image():
#             # 这是根据你前面的报错形状反推出的 ViT-L/14 配置：
#             # hidden=1024, patch=14, pos=257, layers=24
#             return CLIPVisionConfig(
#                 hidden_size=1024,
#                 intermediate_size=4096,
#                 num_hidden_layers=24,
#                 num_attention_heads=16,
#                 image_size=224,
#                 patch_size=14,
#                 projection_dim=768,
#             )
#         else:
#             return CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def load_model(self):
#         if self._is_languagebind_image():
#             clip_l14_cfg = CLIPVisionConfig(
#                 hidden_size=1024,
#                 intermediate_size=4096,
#                 num_hidden_layers=24,
#                 num_attention_heads=16,
#                 image_size=224,
#                 patch_size=14,
#                 projection_dim=768,
#             )

#             self.image_processor = CLIPImageProcessor(
#                 do_resize=True,
#                 size={"shortest_edge": 224},
#                 do_center_crop=True,
#                 crop_size={"height": 224, "width": 224},
#                 do_rescale=True,
#                 rescale_factor=1/255,
#                 do_normalize=True,
#                 image_mean=[0.48145466, 0.4578275, 0.40821073],
#                 image_std=[0.26862954, 0.26130258, 0.27577711],
#             )

#             self.vision_tower = CLIPVisionModel.from_pretrained(
#                 self.vision_tower_name,
#                 config=clip_l14_cfg,
#             )
#         else:
#             self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#             self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == "patch":
#             image_features = image_features[:, 1:]
#         elif self.select_feature == "cls_patch":
#             image_features = image_features
#         else:
#             raise ValueError(f"Unexpected select feature: {self.select_feature}")
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if isinstance(images, list):
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(
#                     image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
#                     output_hidden_states=True
#                 )
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(
#                 images.to(device=self.device, dtype=self.dtype),
#                 output_hidden_states=True
#             )
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype if self.is_loaded else torch.float16

#     @property
#     def device(self):
#         return self.vision_tower.device if self.is_loaded else torch.device("cpu")

#     @property
#     def config(self):
#         return self.vision_tower.config if self.is_loaded else self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2
    
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
