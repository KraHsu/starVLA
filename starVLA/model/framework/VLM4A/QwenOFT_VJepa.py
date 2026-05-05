# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from starVLA.dataloader.vjepa_cache import VJepaFeatureCache
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT, QwenOFTDefaultConfig
from starVLA.model.modules.projector.vjepa_projector import VJepaProjector
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.share_tools import merge_framework_config


@dataclass
class QwenOFTVJepaDefaultConfig(QwenOFTDefaultConfig):
    name: str = "QwenOFT_VJepa"
    vjepa: dict = field(
        default_factory=lambda: {
            "cache_dir": "playground/cache/vjepa/vjepa2_1_vit_b_384/libero_goal",
            "input_dim": 768,
            "projector_hidden_dim": 2048,
            "use_film_gating": True,
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
        self.vjepa_projector = VJepaProjector(
            input_dim=int(vjepa_cfg.input_dim),
            output_dim=int(self.qwen_vl_interface.model.config.hidden_size),
            hidden_dim=int(vjepa_cfg.get("projector_hidden_dim", self.qwen_vl_interface.model.config.hidden_size)),
            use_film_gating=bool(vjepa_cfg.get("use_film_gating", True)),
        )

    def _build_vlm(self):
        from starVLA.model.modules.vlm import get_vlm_model

        return get_vlm_model(config=self.config)

    def _build_action_model(self):
        from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model

        return get_action_model(config=self.config)

    def _load_vjepa_pooled(self, examples: List[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pooled = []
        for example in examples:
            feat = self.vjepa_cache.get_for_sample(example, include_tokens=False)
            pooled.append(feat["pooled"])
        pooled_np = np.stack(pooled, axis=0)
        return torch.from_numpy(pooled_np).to(device=device, dtype=dtype)

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
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
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

        return pred_actions.detach().cpu().float().numpy()
