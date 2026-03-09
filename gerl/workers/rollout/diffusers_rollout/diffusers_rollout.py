# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Rollout with diffusers models.
"""

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Generator, Optional

import numpy as np
import ray
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.profiler import GPUMemoryLogger, simple_timer
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType

from ....protocol import DataProto
from ....utils.lora import select_lora_modules
from ...config import DiffusersModelConfig, DiffusionRolloutConfig
from ...diffusers_model import (
    inject_SDE_scheduler_into_pipeline,
    load_to_device,
    prepare_pipeline,
)
from ..base import BaseRollout
from .utils import get_negative_prompt_embedding

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

__all__ = ["DiffusersSyncRollout", "DiffusersAsyncRollout"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersSyncRollout(BaseRollout):
    def __init__(
        self,
        config: DiffusionRolloutConfig,
        model_config: DiffusersModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.dtype = PrecisionType.to_dtype(self.config.dtype)

        self.pipeline = self._init_diffusion_pipeline(self.dtype)
        self._cached_prompt_embeds: Optional[dict[str, torch.Tensor]] = None

    def _init_diffusion_pipeline(self, dtype) -> "DiffusionPipeline":
        from diffusers import DiffusionPipeline

        local_path = copy_to_local(
            self.model_config.path, use_shm=self.model_config.use_shm
        )
        pipeline = DiffusionPipeline.from_pretrained(local_path)
        pipeline.set_progress_bar_config(disable=True)
        inject_SDE_scheduler_into_pipeline(
            pipeline, pretrained_model_name_or_path=local_path
        )
        prepare_pipeline(pipeline, dtype)

        if self.model_config.use_fused_kernels:
            pipeline.fuse_qkv_projections()

        if self.model_config.lora_rank > 0:
            self._inject_lora(pipeline)

        if self.model_config.use_torch_compile:
            self._compile(pipeline)

        if not self.config.free_cache_engine:
            load_to_device(pipeline, get_device_name())
        return pipeline

    def _inject_lora(self, pipeline: "DiffusionPipeline"):
        from peft import LoraConfig

        assert self.model_config.target_modules is not None

        # Convert config to regular Python types before creating PEFT model
        lora_config = {
            "r": self.model_config.lora_rank,
            "lora_alpha": self.model_config.lora_alpha,
            "init_lora_weights": self.model_config.lora_init_weights,
            "target_modules": convert_to_regular_types(
                select_lora_modules(
                    model_name=os.path.basename(self.model_config.path),
                    target_modules=self.model_config.target_modules,
                )
            ),
            "exclude_modules": convert_to_regular_types(
                self.model_config.exclude_modules
            ),
            "bias": "none",
        }
        pipeline.transformer.add_adapter(LoraConfig(**lora_config))

    def _compile(self, pipeline):
        # compile the transformer modules only.
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, fullgraph=True)
        except Exception as e:
            logger.warning(
                f"Failed to torch.compile the model: {e}, rolling back to eager mode."
            )

    @GPUMemoryLogger(role="diffusers rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        reward_fn = prompts.meta_info.pop("reward_fn", None)
        if self.config.with_reward and reward_fn is None:
            raise ValueError(
                "reward_fn must be provided in meta_info for reward computation."
            )

        if self._cached_prompt_embeds is None:
            self._cached_prompt_embeds = self.cache_prompt_embeds()
        negative_prompt_embeds = self._cached_prompt_embeds[
            "negative_prompt_embeds"
        ].to(get_device_name())
        negative_pooled_prompt_embeds = self._cached_prompt_embeds[
            "negative_pooled_prompt_embeds"
        ].to(get_device_name())

        micro_batches = prompts.split(self.config.log_prob_micro_batch_size_per_gpu)
        generated_input_texts = []
        generated_results = []
        reward_tensors = []
        reward_extra_infos_dicts = defaultdict(list)
        future_rewards = []

        seed = prompts.meta_info.get("seed", None)
        if seed is not None:
            generator = torch.Generator(device=get_device_name()).manual_seed(seed)
        else:
            generator = None

        for micro_batch in micro_batches:
            input_texts = micro_batch.non_tensor_batch["prompt"].tolist()

            noise_level = micro_batch.meta_info.get(
                "noise_level", self.config.noise_level
            )
            num_inference_steps = micro_batch.meta_info.get(
                "num_inference_steps", self.config.num_inference_steps
            )

            output = self.pipeline(
                input_texts,
                height=self.config.image_height,
                width=self.config.image_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                max_sequence_length=self.config.prompt_length,
                negative_prompt_embeds=negative_prompt_embeds.repeat(
                    len(input_texts), 1, 1
                ),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.repeat(
                    len(input_texts), 1
                ),
                output_type="pt",
                noise_level=noise_level,
                sde_window_size=self.config.sde_window_size,
                sde_window_range=self.config.sde_window_range,
                sde_type=self.config.sde_type,
                rollout_solver=self.config.rollout_solver,
            )

            result = TensorDict(
                {
                    "responses": output.images,
                    "latents": output.all_latents,
                    "rollout_log_probs": output.all_log_probs,
                    "old_preds": output.all_preds,
                    "timesteps": output.all_timesteps,
                    "prompt_embeds": output.prompt_embeds,
                    "pooled_prompt_embeds": output.pooled_prompt_embeds,
                    "negative_prompt_embeds": output.negative_prompt_embeds,
                    "negative_pooled_prompt_embeds": output.negative_pooled_prompt_embeds,
                },
                batch_size=len(output.images),
            )
            # launch async micro_batch reward computing
            if self.config.with_reward:
                future_reward = reward_fn.remote(
                    data=DataProto(
                        batch=result.select("responses"),
                        non_tensor_batch=micro_batch.non_tensor_batch,
                    )
                )
                future_rewards.append(future_reward)

            # concatenate generation result
            generated_results.append(result)
            generated_input_texts.extend(input_texts)

        result = DataProto(
            batch=torch.cat(generated_results),
            non_tensor_batch={"prompt": np.array(generated_input_texts)},
        )
        result.meta_info["cached_steps"] = result.batch["timesteps"].shape[1]

        if self.config.with_reward:
            timing_reward: dict[str, float] = {}
            with simple_timer("reward", timing_reward):
                # concatenate reward result batches
                for reward_tensor, reward_extra_infos_dict in ray.get(future_rewards):
                    reward_tensors.append(reward_tensor)
                    for k, v in reward_extra_infos_dict.items():
                        reward_extra_infos_dicts[k].extend(v)
            result.meta_info["timing_reward"] = timing_reward

            # we combine with rewards in result
            result.batch["instance_level_scores"] = torch.cat(reward_tensors)
            if reward_extra_infos_dicts:
                result.non_tensor_batch.update(
                    {k: np.array(v) for k, v in reward_extra_infos_dicts.items()}
                )
            result.batch["instance_level_rewards"] = result.batch[
                "instance_level_scores"
            ]

        return result

    def cache_prompt_embeds(
        self, negative_prompt: str = ""
    ) -> dict[str, torch.FloatTensor]:
        embedding = get_negative_prompt_embedding(
            self.pipeline,
            negative_prompt=negative_prompt,
            max_sequence_length=self.config.prompt_length,
        )
        return {k: v.to("cpu") for k, v in embedding.items()}

    async def resume(self):
        """Resume rollout weights in GPU memory."""
        load_to_device(self.pipeline, get_device_name())

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        if self.model_config.lora_rank > 0:
            return self.update_lora_weights(weights)
        else:
            return self.update_full_weights(weights)

    async def release(self):
        """Release rollout weights in GPU memory."""
        load_to_device(self.pipeline, "cpu")

    def update_lora_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ):
        """Update the LoRA weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        from .utils import load_peft_weight_from_state_dict

        load_peft_weight_from_state_dict(self.pipeline.transformer, dict(weights))

    def update_full_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ):
        """Update the full weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        is_compiled = hasattr(self.pipeline.transformer, "_orig_mod")

        state_dict = self.pipeline.transformer.state_dict()
        for name, tensor in weights:
            if is_compiled:
                name = "_orig_mod." + name

            if name in state_dict:
                if is_compiled:
                    # to prevent recompilation, we need to use unsafe .data.copy operation
                    if state_dict[name].shape == tensor.shape:
                        state_dict[name].data.copy_(tensor.data)
                    else:
                        raise ValueError(
                            f"Cannot load weights, shape mismatch for {name}: "
                            f"{state_dict[name].shape} vs {tensor.shape}."
                        )
                else:
                    state_dict[name].copy_(tensor)
            else:
                logger.warning(f"Parameter {name} not found in model state_dict.")


class DiffusersAsyncRollout(DiffusersSyncRollout):
    """
    Async rollout currently shares the same implementation as DiffusersSyncRollout.
    This class exists for future extension; a full async implementation may be added later.
    """

    ...
