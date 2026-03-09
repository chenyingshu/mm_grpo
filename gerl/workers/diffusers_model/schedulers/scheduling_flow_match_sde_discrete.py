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


from dataclasses import dataclass
from typing import Literal, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput

from ..solvers.solver import sample_previous_step_by_solver

@dataclass
class FlowMatchSDEDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor
    log_prob: torch.FloatTensor
    prev_sample_mean: torch.FloatTensor
    std_dev_t: torch.FloatTensor


class FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None,
        sde_type: Literal["sde", "cps"] = "sde",
        rollout_solver: Literal["sde"] = "sde",
    ) -> FlowMatchSDEDiscreteSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Modified from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if prev_sample is not None:
            prev_sample = prev_sample.to(torch.float32)

        prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sample_previous_step(
            sample=sample,
            model_output=model_output,
            generator=generator,
            per_token_timesteps=per_token_timesteps,
            noise_level=noise_level,
            prev_sample=prev_sample,
            sde_type=sde_type,
            rollout_solver=rollout_solver,
        )

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample, log_prob, prev_sample_mean, std_dev_t)
        return FlowMatchSDEDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            log_prob=log_prob,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
        )

    def sample_previous_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.Tensor] = None,
        sde_type: Literal["cps", "sde"] = "sde",
        rollout_solver: Literal["sde"] = "sde",
    ):
        # check inputs
        assert sample.dtype == torch.float32
        if prev_sample is not None:
            assert prev_sample.dtype == torch.float32

        if per_token_timesteps is not None:
            raise NotImplementedError(
                "per_token_timesteps is not supported yet for FlowMatchSDEDiscreteScheduler."
            )
        else:
            if timestep is None:
                sigma_idx = self.step_index
                sigma = self.sigmas[sigma_idx]
                sigma_next = self.sigmas[sigma_idx + 1]
            else:
                sigma_idx = torch.tensor([self.index_for_timestep(t) for t in timestep])
                sigma = self.sigmas[sigma_idx].view(
                    -1, *([1] * (len(sample.shape) - 1))
                )
                sigma_next = self.sigmas[sigma_idx + 1].view(
                    -1, *([1] * (len(sample.shape) - 1))
                )

            sigma_max = self.sigmas[1]

        solver_kwargs: dict = {
            "rollout_solver": rollout_solver,
            "sample": sample,
            "model_output": model_output,
            "sigma": sigma,
            "sigma_next": sigma_next,
            "sigma_max": sigma_max,
            "noise_level": noise_level,
            "prev_sample": prev_sample,
            "sde_type": sde_type,
            "generator": generator,
        }

        prev_sample, log_prob, prev_sample_mean, std_dev_t = sample_previous_step_by_solver(
            **solver_kwargs,
        )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t
