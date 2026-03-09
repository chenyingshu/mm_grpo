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
Single Process Actor
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.actor import BasePPOActor

from ...protocol import DataProto
from ...trainer.ppo.core_algos import get_kl_loss_fn, get_policy_loss_fn
from ...utils.ema import EMAModuleWrapper
from ..config import DiffusionFSDPActorConfig

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = ["DiffusersPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersPPOActor(BasePPOActor):
    """Diffusers PPO Actor

    Args:
        config (DiffusionActorConfig): Configuration for the actor
        actor_module (nn.Module): The actor module
        scheduler (SchedulerMixin): The scheduler for diffusion process
        actor_optimizer (Optional[torch.optim.Optimizer], optional): The optimizer for the actor.
            When None, it is Reference Policy. Defaults to None.
        ema_wrapper (Optional[EMAModuleWrapper], optional): EMA wrapper for the actor module.
            Defaults to None.
    """

    def __init__(
        self,
        config: DiffusionFSDPActorConfig,
        actor_module: nn.Module,
        scheduler: "SchedulerMixin",
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        ema_wrapper: Optional[EMAModuleWrapper] = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.ema_wrapper = ema_wrapper
        self.scheduler = scheduler
        self.scheduler.set_timesteps(
            config.num_inference_steps, device=get_device_name()
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(
            self.config.fsdp_config.get("dtype", "bfloat16")
        )
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400, init_scale=2**24)
        else:
            self.scaler = None

    def _predict_micro_batch(
            self, micro_batch: dict[str, torch.Tensor], step: int
    ) -> torch.Tensor:
        """
        Returns:
            noise_pred: single step noise predction, shape (bs, )
        """

        latents = micro_batch["latents"]
        timesteps = micro_batch["timesteps"]
        prompt_embeds = micro_batch["prompt_embeds"]
        pooled_prompt_embeds = micro_batch["pooled_prompt_embeds"]
        negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
        negative_pooled_prompt_embeds = micro_batch["negative_pooled_prompt_embeds"]

        if self.config.guidance_scale > 1.0:
            noise_pred = self.actor_module(
                hidden_states=torch.cat([latents[:, step]] * 2),
                timestep=torch.cat([timesteps[:, step]] * 2),
                encoder_hidden_states=torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                ),
                pooled_projections=torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                ),
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = self.actor_module(
                hidden_states=latents[:, step],
                timestep=timesteps[:, step],
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

        return noise_pred

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: # (bs, )
        """

        latents = micro_batch["latents"]
        timesteps = micro_batch["timesteps"]
        prompt_embeds = micro_batch["prompt_embeds"]
        pooled_prompt_embeds = micro_batch["pooled_prompt_embeds"]
        negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
        negative_pooled_prompt_embeds = micro_batch["negative_pooled_prompt_embeds"]

        if self.config.guidance_scale > 1.0:
            noise_pred = self.actor_module(
                hidden_states=torch.cat([latents[:, step]] * 2),
                timestep=torch.cat([timesteps[:, step]] * 2),
                encoder_hidden_states=torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                ),
                pooled_projections=torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                ),
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = self.actor_module(
                hidden_states=latents[:, step],
                timestep=timesteps[:, step],
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

        _, log_prob, prev_sample_mean, std_dev_t = self.scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred,
            timestep=timesteps[:, step],
            noise_level=self.config.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=self.config.sde_type,
        )

        return log_prob, prev_sample_mean, std_dev_t

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        assert self.actor_optimizer is not None

        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(
                    f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}"
                )
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def compute_log_prob(self, data: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the log probability and previous sample mean for each action in the data."""
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = [
            "latents",
            "timesteps",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]
        data = data.select(batch_keys=select_keys)
        micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        prev_sample_mean_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            log_probs_lst_steps = []
            prev_sample_mean_lst_steps = []
            for step in range(micro_batch.meta_info["cached_steps"]):
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                with torch.no_grad():
                    log_probs, prev_sample_mean, _ = self._forward_micro_batch(
                        model_inputs, step=step
                    )
                log_probs_lst_steps.append(log_probs)
                prev_sample_mean_lst_steps.append(prev_sample_mean)
            log_probs_lst_steps = torch.stack(log_probs_lst_steps, dim=1)
            prev_sample_mean_lst_steps = torch.stack(prev_sample_mean_lst_steps, dim=1)
            log_probs_lst.append(log_probs_lst_steps)
            prev_sample_mean_lst.append(prev_sample_mean_lst_steps)

        log_probs = torch.concat(log_probs_lst, dim=0)
        prev_sample_mean = torch.concat(prev_sample_mean_lst, dim=0)

        return log_probs, prev_sample_mean

    def _compute_nft_old_and_forward_on_same_xt(
        self,
        model_inputs: dict[str, torch.Tensor],
        step: int,
        compute_ref: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        peft_model = getattr(
            self.actor_module, "_fsdp_wrapped_module", self.actor_module
        )
        assert (
            hasattr(peft_model, "peft_config")
            and "old" in getattr(peft_model, "peft_config", {})
        ), "DiffusionNFT old adapter is missing. Expected adapter 'old' for same-xt old prediction."

        peft_model.set_adapter("old")
        try:
            with torch.no_grad():
                old_prediction = self._predict_micro_batch(model_inputs, step=step)
        finally:
            peft_model.set_adapter("default")

        forward_prediction = self._predict_micro_batch(model_inputs, step=step)

        ref_prediction: Optional[torch.Tensor] = None
        if compute_ref:
            peft_model.disable_adapters()
            try:
                with torch.no_grad():
                    ref_prediction = self._predict_micro_batch(model_inputs, step=step)
            finally:
                peft_model.enable_adapters()

        return old_prediction, forward_prediction, ref_prediction

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def update_policy(self, data: DataProto):
        assert self.actor_optimizer is not None

        loss_mode = getattr(
            self.config.policy_loss, "loss_mode", "flow_grpo"
        )

        # make sure we are in training mode
        self.actor_module.train()

        select_keys = [
            "latents",
            "advantages",
            "timesteps",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]
        if loss_mode == "flow_grpo":
            select_keys.append("old_log_probs")
        if self.config.use_kl_loss and loss_mode != "diffusion_nft":
            select_keys.append("ref_prev_sample_mean")

        data = data.select(batch_keys=select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics: dict[str, Any] = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.shuffle_micro_batch:
                    mini_batch.reorder(torch.randperm(len(mini_batch)))

                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size
                    * data.meta_info["cached_steps"]
                    // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    for step in range(micro_batch.meta_info["cached_steps"]):
                        micro_batch_metrics = {}
                        model_inputs = {
                            **micro_batch.batch,
                            **micro_batch.non_tensor_batch,
                        }
                        advantages = model_inputs["advantages"]

                        loss_scale_factor = 1 / self.gradient_accumulation
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        if loss_mode == "flow_grpo":
                            old_log_prob = model_inputs["old_log_probs"]
                            log_prob, prev_sample_mean, std_dev_t = (
                                self._forward_micro_batch(model_inputs, step=step)
                            )

                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob[:, step],
                                log_prob=log_prob,
                                advantages=advantages,
                                config=self.config,
                            )
                        elif loss_mode == "diffusion_nft":
                            x0 = model_inputs["latents"][:, -1]
                            t = model_inputs["timesteps"][:, step] / 1000.0
                            t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
                            noise = torch.randn_like(x0.float())
                            xt = (1 - t_expanded) * x0 + t_expanded * noise
                            model_inputs["latents"][:, step] = xt
                            old_prediction, forward_prediction, ref_prediction = (
                                self._compute_nft_old_and_forward_on_same_xt(
                                    model_inputs=model_inputs,
                                    step=step,
                                    compute_ref=self.config.use_kl_loss,
                                )
                            )

                            pg_loss, pg_metrics = policy_loss_fn(
                                x0=x0,
                                xt=xt,
                                t_expanded=t_expanded,
                                old_prediction=old_prediction,
                                forward_prediction=forward_prediction,
                                advantages=advantages,
                                config=self.config,
                            )
                        else:
                            raise ValueError(f"Unknown loss mode: {loss_mode}")
                        micro_batch_metrics.update(pg_metrics)

                        policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            kl_fn = get_kl_loss_fn(loss_mode)
                            if loss_mode == "flow_grpo":
                                kl_loss, kl_metrics = kl_fn(
                                    prev_sample_mean=prev_sample_mean,
                                    ref_prev_sample_mean=model_inputs[
                                        "ref_prev_sample_mean"
                                    ][:, step],
                                    std_dev_t=std_dev_t,
                                )
                            elif loss_mode == "diffusion_nft":
                                kl_loss, kl_metrics = kl_fn(
                                    forward_prediction=forward_prediction,
                                    ref_prediction=ref_prediction,
                                )
                            else:
                                raise ValueError(
                                    f"No KL loss registered for: {loss_mode}"
                                )

                            policy_loss = (
                                policy_loss + kl_loss * self.config.kl_loss_coef
                            )
                            micro_batch_metrics.update(kl_metrics)
                            micro_batch_metrics["actor/kl_loss"] = (
                                kl_loss.detach().item() * loss_scale_factor
                            )
                            micro_batch_metrics["actor/kl_coef"] = (
                                self.config.kl_loss_coef
                            )

                        loss = policy_loss * loss_scale_factor
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        micro_batch_metrics["actor/pg_loss"] = (
                            pg_loss.detach().item() * loss_scale_factor
                        )
                        append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if self.ema_wrapper is not None:
                    self.ema_wrapper.step(self.actor_module.parameters())

                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
