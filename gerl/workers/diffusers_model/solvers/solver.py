import math
from typing import Literal, Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

def _batch_reduce(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=tuple(range(1, value.ndim)))


def sample_with_sde_solver(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_max: torch.Tensor,
    noise_level: float,
    prev_sample: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    sde_type: Literal["cps", "sde"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = sigma_next - sigma
    if sde_type == "sde":
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
            * noise_level
        )
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    elif sde_type == "cps":
        std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_next) + noise_estimate * torch.sqrt(
            sigma_next**2 - std_dev_t**2
        )
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    return prev_sample, _batch_reduce(log_prob), prev_sample_mean, std_dev_t


def sample_previous_step_by_solver(
    rollout_solver: Literal["sde"],
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_max: torch.Tensor,
    noise_level: float,
    prev_sample: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    sde_type: Literal["cps", "sde"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rollout_solver == "sde":
        return sample_with_sde_solver(
            sample=sample,
            model_output=model_output,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
            noise_level=noise_level,
            prev_sample=prev_sample,
            generator=generator,
            sde_type=sde_type,
        )
    raise ValueError(f"Unsupported rollout_solver: {rollout_solver}")
