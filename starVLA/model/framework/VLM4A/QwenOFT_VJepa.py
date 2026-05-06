# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from contextlib import nullcontext
import os
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.vjepa_cache import VJepaFeatureCache
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT, QwenOFTDefaultConfig
from starVLA.model.modules.projector.vjepa_projector import VJepaProjector
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.training.trainer_utils.trainer_tools import resize_images

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VJEPA2_ROOT = _REPO_ROOT / "third_party" / "vjepa2"
if _VJEPA2_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _VJEPA2_ROOT.as_posix())

MODEL_CHECKPOINT_KEY = "ema_encoder"
MODEL_CHECKPOINT_FALLBACK_KEYS = ("encoder", "target_encoder")
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


@dataclass
class QwenOFTVJepaDefaultConfig(QwenOFTDefaultConfig):
    name: str = "QwenOFT_VJepa"
    vjepa: dict = field(
        default_factory=lambda: {
            "cache_dir": "playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal",
            "input_dim": 768,
            "projector_hidden_dim": 2048,
            "fusion": "film_gating",
            "use_film_gating": True,
            "num_attention_heads": 8,
            "online_eval_fallback": True,
            "encoder_ckpt_path": None,
            "num_history_frames": 8,
            "img_size": 384,
            "dtype": "bf16",
            "primary_camera_index": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT_VJepa")
class QwenOFT_VJepa(Qwenvl_OFT):
    """QwenOFT augmented with offline V-JEPA pooled features."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenOFTVJepaDefaultConfig, config)
        self.qwen_vl_interface = self._build_vlm()
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = self._build_action_model()

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        self.l1_loss = nn.L1Loss()

        vjepa_cfg = self.config.framework.vjepa
        self.vjepa_cache_dir = Path(vjepa_cfg.cache_dir)
        self.vjepa_cache = VJepaFeatureCache(self.vjepa_cache_dir, include_tokens=False)
        self.vjepa_online_enabled = bool(vjepa_cfg.get("online_eval_fallback", True))
        self.vjepa_online_encoder = None
        self.vjepa_frame_history = deque(maxlen=int(vjepa_cfg.get("num_history_frames", 8)))
        self.vjepa_num_history_frames = int(vjepa_cfg.get("num_history_frames", 8))
        self.vjepa_img_size = int(vjepa_cfg.get("img_size", 384))
        self.vjepa_dtype_name = str(vjepa_cfg.get("dtype", "bf16"))
        self.vjepa_primary_camera_index = int(vjepa_cfg.get("primary_camera_index", 0))
        self.vjepa_encoder_ckpt_path = self._resolve_vjepa_encoder_ckpt_path(vjepa_cfg)
        self.vjepa_online_transform = None
        self.vjepa_projector = VJepaProjector(
            input_dim=int(vjepa_cfg.input_dim),
            output_dim=int(self.qwen_vl_interface.model.config.hidden_size),
            hidden_dim=int(vjepa_cfg.get("projector_hidden_dim", self.qwen_vl_interface.model.config.hidden_size)),
            fusion=str(vjepa_cfg.get("fusion", "film_gating")),
            num_attention_heads=int(vjepa_cfg.get("num_attention_heads", 8)),
            use_film_gating=bool(vjepa_cfg.get("use_film_gating", True)),
        )

    def reset(self, **kwargs) -> None:
        self.vjepa_frame_history.clear()
        self._vjepa_last_task_id = None

    def _build_vlm(self):
        from starVLA.model.modules.vlm import get_vlm_model

        return get_vlm_model(config=self.config)

    def _build_action_model(self):
        from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model

        return get_action_model(config=self.config)

    def _load_vjepa_pooled(self, examples: List[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if all(self._sample_has_cache_key(example) for example in examples):
            return self._load_vjepa_pooled_from_cache(examples, device=device, dtype=dtype)
        if not self.vjepa_online_enabled:
            raise KeyError(
                "V-JEPA examples are missing episode_id/frame_id and online_eval_fallback is disabled."
            )
        return self._load_vjepa_pooled_online(examples, device=device, dtype=dtype)

    @staticmethod
    def _sample_has_cache_key(example: dict) -> bool:
        return (example.get("episode_id", example.get("trajectory_id")) is not None) and example.get("frame_id") is not None

    def _load_vjepa_pooled_from_cache(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pooled = []
        for example in examples:
            feat = self.vjepa_cache.get_for_sample(example, include_tokens=False)
            pooled.append(feat["pooled"])
        pooled_np = np.stack(pooled, axis=0)
        return torch.from_numpy(pooled_np).to(device=device, dtype=dtype)

    def _resolve_vjepa_encoder_ckpt_path(self, vjepa_cfg) -> str | None:
        env_ckpt_path = os.environ.get("VJEPA_ENCODER_CKPT")
        if env_ckpt_path:
            return os.fspath(env_ckpt_path)
        ckpt_path = vjepa_cfg.get("encoder_ckpt_path", None)
        if ckpt_path:
            return os.fspath(ckpt_path)
        index = getattr(self.vjepa_cache, "index", {})
        ckpt_path = index.get("checkpoint_path")
        if ckpt_path:
            return os.fspath(ckpt_path)
        return None

    def _load_vjepa_pooled_online(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        encoder = self._get_online_vjepa_encoder(device)
        encoder_dtype = next(encoder.parameters()).dtype
        clips = [self._build_online_clip(example) for example in examples]
        batch = torch.stack(clips, dim=0).to(device=device, dtype=encoder_dtype, non_blocking=(device.type == "cuda"))

        use_autocast = device.type == "cuda" and encoder_dtype in {torch.bfloat16, torch.float16}
        ctx = torch.autocast(device_type="cuda", dtype=encoder_dtype) if use_autocast else nullcontext()
        with ctx:
            tokens = encoder(batch)
        pooled = tokens.float().mean(dim=1)
        return pooled.to(device=device, dtype=dtype)

    def _get_online_vjepa_transform(self):
        if getattr(self, "vjepa_online_transform", None) is None:
            from evals.video_classification_frozen.utils import make_transforms

            self.vjepa_online_transform = make_transforms(training=False, crop_size=self.vjepa_img_size)
        return self.vjepa_online_transform

    def _get_online_vjepa_encoder(self, device: torch.device):
        if self.vjepa_online_encoder is not None:
            return self.vjepa_online_encoder
        if self.vjepa_encoder_ckpt_path is None:
            raise ValueError(
                "Cannot run online V-JEPA eval fallback without framework.vjepa.encoder_ckpt_path "
                "or checkpoint_path in cache index.json."
            )
        if self.vjepa_num_history_frames < 1 or self.vjepa_num_history_frames % 2 != 0:
            raise ValueError("framework.vjepa.num_history_frames must be a positive even integer")

        self._patch_vjepa2_1_rope_dtype()
        from app.vjepa_2_1.models import vision_transformer as vit

        encoder = vit.vit_base(
            patch_size=16,
            img_size=(self.vjepa_img_size, self.vjepa_img_size),
            num_frames=self.vjepa_num_history_frames,
            tubelet_size=2,
            use_sdpa=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        checkpoint = torch.load(self.vjepa_encoder_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = None
        for key in (MODEL_CHECKPOINT_KEY, *MODEL_CHECKPOINT_FALLBACK_KEYS):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            raise KeyError(
                f"Could not find one of checkpoint keys {(MODEL_CHECKPOINT_KEY, *MODEL_CHECKPOINT_FALLBACK_KEYS)} "
                f"in {self.vjepa_encoder_ckpt_path}"
            )
        state_dict = self._strip_vjepa_prefix(state_dict)
        msg = encoder.load_state_dict(state_dict, strict=True)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                f"Unexpected V-JEPA encoder load result for {self.vjepa_encoder_ckpt_path}: "
                f"missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
            )
        encoder = encoder.to(device=device, dtype=self._resolve_vjepa_dtype(device))
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
        self.vjepa_online_encoder = encoder
        return self.vjepa_online_encoder

    def _resolve_vjepa_dtype(self, device: torch.device) -> torch.dtype:
        if device.type == "cpu":
            return torch.float32
        if self.vjepa_dtype_name == "fp16":
            return torch.float16
        if self.vjepa_dtype_name == "fp32":
            return torch.float32
        return torch.bfloat16

    @staticmethod
    def _strip_vjepa_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()}

    @staticmethod
    def _patch_vjepa2_1_rope_dtype() -> None:
        from app.vjepa_2_1.models.utils import modules as vjepa_modules

        if getattr(vjepa_modules.rotate_queries_or_keys, "_starvla_dtype_safe", False):
            return

        original_rotate = vjepa_modules.rotate_queries_or_keys

        def dtype_safe_rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
            return original_rotate(x, pos, n_registers, has_cls_first).to(dtype=x.dtype)

        dtype_safe_rotate_queries_or_keys._starvla_dtype_safe = True
        dtype_safe_rotate_queries_or_keys._starvla_original = original_rotate
        vjepa_modules.rotate_queries_or_keys = dtype_safe_rotate_queries_or_keys

    def _build_online_clip(self, example: dict) -> torch.Tensor:
        image_obj = to_pil_preserve(example.get("vjepa_image", example["image"]))
        if isinstance(image_obj, (list, tuple)):
            if not image_obj:
                raise ValueError("example['image'] is empty; cannot build online V-JEPA clip")
            image_obj = image_obj[min(self.vjepa_primary_camera_index, len(image_obj) - 1)]
        if not isinstance(image_obj, Image.Image):
            raise TypeError(f"Expected a PIL image after conversion, got {type(image_obj)}")

        task_id = str(example.get("lang", ""))
        if bool(example.get("vjepa_reset_history", False)) or getattr(self, "_vjepa_last_task_id", None) != task_id:
            self.vjepa_frame_history.clear()
            self._vjepa_last_task_id = task_id
        self.vjepa_frame_history.append(image_obj.convert("RGB"))

        frames = list(self.vjepa_frame_history)
        if not frames:
            raise ValueError("No frames available for online V-JEPA fallback")
        while len(frames) < self.vjepa_num_history_frames:
            frames.insert(0, frames[0])
        frames = frames[-self.vjepa_num_history_frames :]

        clip = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
        transformed = self._preprocess_online_clip(clip)
        if transformed.shape[1] != self.vjepa_num_history_frames:
            raise RuntimeError(
                f"Online V-JEPA clip has {transformed.shape[1]} frames, expected {self.vjepa_num_history_frames}"
            )
        return transformed

    def _preprocess_online_clip(self, clip: np.ndarray) -> torch.Tensor:
        transform = self._get_online_vjepa_transform()
        transformed = transform(clip)
        if not isinstance(transformed, list) or len(transformed) != 1:
            raise RuntimeError(f"Unexpected V-JEPA transform output type: {type(transformed)}")
        return transformed[0]

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pooled = self._load_vjepa_pooled(
                examples, device=action_queries.device, dtype=action_queries.dtype
            )
            projected = self.vjepa_projector(pooled)
            action_queries = self.vjepa_projector.apply_to_queries(projected, action_queries)

            pred_actions = self.action_model.predict_action(action_queries)
            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        datasets_cfg = getattr(getattr(self, "config", None), "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None)
        train_obs_image_size = getattr(vla_data_cfg, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pooled = self._load_vjepa_pooled(
                examples, device=action_queries.device, dtype=action_queries.dtype
            )
            projected = self.vjepa_projector(pooled)
            action_queries = self.vjepa_projector.apply_to_queries(projected, action_queries)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().float().numpy()
        return {"normalized_actions": normalized_actions}
