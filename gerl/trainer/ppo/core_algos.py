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
Core functions to implement FlowGRPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = [
    "register_adv_est",
    "get_adv_estimator_fn",
    "AdvantageEstimator",
    "register_kl_loss",
    "get_kl_loss_fn",
]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        Optional[DictConfig],  # config
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

KlLossFn = Callable[..., tuple[torch.Tensor, dict[str, Any]]]

KL_LOSS_REGISTRY: dict[str, KlLossFn] = {}

def register_kl_loss(name: str) -> Callable[[KlLossFn], KlLossFn]:
    """Register a KL loss function with the given name.
    Args:
        name (str): The name to register the KL loss function under.
    Returns:
        function: Decorator function that registers the KL loss function.
    """

    def decorator(func: KlLossFn) -> KlLossFn:
        KL_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_kl_loss_fn(name: str) -> KlLossFn:
    """Get the KL loss function with a given name.
    Args:
        name: The name of the KL loss function.
    Returns:
        The KL loss function.
    """
    if name not in KL_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported KL loss: {name}. Supported: {list(KL_LOSS_REGISTRY.keys())}"
        )
    return KL_LOSS_REGISTRY[name]

def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `gerl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    FLOW_GRPO = "flow_grpo"  # newly added for diffusion models


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


@register_adv_est(AdvantageEstimator.FLOW_GRPO)
def compute_flow_grpo_outcome_advantage(
    instance_level_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-4,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DictConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        instance_level_rewards: `(torch.Tensor)`
            shape is (bs, )
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        global_std: `(bool)`
            whether to use global std for advantage normalization
        config: `(Optional[DictConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, )
        Returns: `(torch.Tensor)`
            shape is (bs, )
    """
    scores = instance_level_rewards

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        if global_std:
            batch_std = torch.std(scores)
        else:
            batch_std = None

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores


@register_policy_loss("flow_grpo")  # type: ignore[arg-type]
def compute_policy_loss_flow_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for FlowGRPO.

    Adapted from
    https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size,).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size,).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size,).
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
    """
    advantages = torch.clamp(
        advantages,
        -config.clip_max,
        config.clip_max,
    )
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    )
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    pg_metrics = {"actor/ppo_kl": policy_loss.detach().item()}
    return policy_loss, pg_metrics

@register_policy_loss("diffusion_nft")  # type: ignore[arg-type]
def compute_policy_loss_diffusion_nft(
    x0: torch.Tensor,
    xt: torch.Tensor,
    t_expanded: torch.Tensor,
    old_prediction: torch.Tensor,
    forward_prediction: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for FlowGRPO.

    Adapted from
    https://github.com/NVlabs/DiffusionNFT/blob/main/scripts/train_nft_sd3.py#L909

    Args:
        x0 (torch.Tensor):
            Clean latent under the old policy, shape (batch_size,).
        xt (torch.Tensor):
            Noisy latent at timestep t, shape (batch_size,).
        t_expanded (torch.Tensor):
            Timestep t, shape (batch_size,).
        old_prediction (torch.Tensor):
            Noise prediction of actions under the old policy, shape (batch_size,).
        forward_prediction (torch.Tensor):
            Noise prediction of actions under the current policy, shape (batch_size,).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size,).
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
    """
    nft_beta = getattr(config.policy_loss, "nft_beta", 1.0)

    # clip advantages
    advantages = torch.clamp(
        advantages,
        -config.clip_max,
        config.clip_max,
    )  # default adv_mode is "all"
    if hasattr(config, "adv_mode"):
        if config.adv_mode == "positive_only":
            advantages = torch.clamp(advantages, 0, config.clip_max)
        elif config.adv_mode == "negative_only":
            advantages = torch.clamp(advantages, -config.clip_max, 0)
        elif config.adv_mode == "one_only":
            advantages = torch.where(
                advantages > 0,
                torch.ones_like(advantages),
                torch.zeros_like(advantages),
            )
        elif config.adv_mode == "binary":
            advantages = torch.sign(advantages)

    # normalize advantage
    normalized_advantages_clip = (advantages / config.clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)
    positive_prediction = (
            nft_beta * forward_prediction
            + (1 - nft_beta) * old_prediction.detach()
    )
    implicit_negative_prediction = (
                                           1.0 + nft_beta
                                   ) * old_prediction.detach() - nft_beta * forward_prediction

    # adaptive weighting
    x0_prediction = xt - t_expanded * positive_prediction
    with torch.no_grad():
        weight_factor = (
            torch.abs(x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
        dim=tuple(range(1, x0.ndim))
    )
    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
    with torch.no_grad():
        negative_weight_factor = (
            torch.abs(negative_x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
        dim=tuple(range(1, x0.ndim))
    )

    ori_policy_loss = (
            r * positive_loss / nft_beta
            + (1.0 - r) * negative_loss / nft_beta
    )
    policy_loss = (ori_policy_loss * config.clip_max).mean()

    pg_metrics = {"actor/ppo_kl": policy_loss.detach().item()}
    return policy_loss, pg_metrics


@register_kl_loss("flow_grpo")
def kl_loss_flow_grpo(
    prev_sample_mean: torch.Tensor,
    ref_prev_sample_mean: torch.Tensor,
    std_dev_t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Gaussian KL used by FlowGRPO (sample-mean space, variance-scaled)."""
    kl = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(
        dim=(1, 2, 3), keepdim=True
    ) / (2 * std_dev_t**2)
    kl_mean = kl.mean()
    return kl_mean, {"actor/kl_div": kl_mean.detach().item()}


@register_kl_loss("diffusion_nft")
def kl_loss_diffusion_nft(
    forward_prediction: torch.Tensor,
    ref_prediction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MSE KL in v-prediction space used by DiffusionNFT.

    Adapted from
    https://github.com/NVlabs/DiffusionNFT/blob/main/scripts/train_nft_sd3.py#L966
    """
    kl = ((forward_prediction - ref_prediction) ** 2).mean(
        dim=tuple(range(1, forward_prediction.ndim))
    )
    kl_mean = kl.mean()
    return kl_mean, {"actor/kl_div": kl_mean.detach().item()}
