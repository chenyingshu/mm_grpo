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
from typing import Literal

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.workers.config.optimizer import OptimizerConfig

from .diffusers_model import DiffusersModelConfig
from .engine import FSDPEngineConfig

__all__ = ["DiffusionActorConfig", "DiffusionFSDPActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'flow_grpo'
        nft_beta (float): DiffusionNFT beta for positive/negative prediction. Used when loss_mode=diffusion_nft.
        nft_decay_type (int): DiffusionNFT old-adapter EMA decay type (0=no decay, 1=uprate 0.001 uphold 0.5, 2=flat 75 uprate 0.0075 uphold 0.999).
    """

    loss_mode: str = "flow_grpo"
    nft_beta: float = 1.0
    nft_decay_type: int = 2


@dataclass
class DiffusionActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        nnodes (int): Number of nodes for actor. (For standalone mode only.)
        n_gpus_per_node (int): Number of GPUs per node for actor. (For standalone mode only.)
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size_per_gpu (int): Micro-batch size per GPU for PPO training.
        shuffle_micro_batch (bool): Whether to shuffle micro-batches during training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_max (float): Maximum absolute value for advantage clipping.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        use_kl_loss (bool): Whether to use KL divergence loss.
        kl_loss_coef (float): KL divergence loss coefficient.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
    }

    strategy: str = MISSING
    nnodes: int = 0
    n_gpus_per_node: int = 0
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 8
    shuffle_micro_batch: bool = False
    clip_ratio: float = 1e-4
    clip_max: float = 5.0
    adv_mode: str = "all"
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.04
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    data_loader_seed = 1
    model_config: DiffusersModelConfig = field(default_factory=DiffusersModelConfig)
    guidance_scale: float = 4.5
    noise_level: float = 0.7
    sde_type: Literal["sde", "cps"] = "sde"
    rollout_solver: Literal["sde"] = "sde"
    num_inference_steps: int = 10

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING


@dataclass
class DiffusionFSDPActorConfig(DiffusionActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
