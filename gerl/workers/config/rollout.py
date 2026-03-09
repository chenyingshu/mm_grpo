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

from dataclasses import dataclass, field
from typing import Literal, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig

__all__ = ["DiffusionRolloutConfig"]


@dataclass
class SamplingConfig(BaseConfig):
    n: int = 1
    noise_level: float = 0.0
    num_inference_steps: int = 40
    seed: int = 42


@dataclass
class DiffusionRolloutConfig(BaseConfig):
    name: Optional[str] = MISSING
    mode: str = "sync"
    nnodes: int = 0
    n_gpus_per_node: int = 0
    with_reward: bool = True

    prompt_length: int = 128
    image_height: int = 512
    image_width: int = 512
    log_prob_micro_batch_size_per_gpu: int = 8
    num_inference_steps: int = 10
    noise_level: float = 0.7
    guidance_scale: float = 4.5
    sde_type: Literal["sde", "cps"] = "sde"
    rollout_solver: Literal["sde"] = "sde"
    sde_window_size: Optional[int] = None
    sde_window_range: Optional[tuple[int, int]] = None

    dtype: str = "bfloat16"
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1

    n: int = 8

    layered_summon: bool = False

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self):
        """Validate the rollout config"""
        pass
