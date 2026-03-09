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

"""DiffusionNFT-specific utilities for old-policy adapter update."""

import torch


def return_decay(step: int, decay_type: int) -> float:
    """Adaptive decay for DiffusionNFT old-policy EMA.
    Args:
        step: Global training step.
        decay_type: 0=no decay, 1=uprate 0.001 uphold 0.5, 2=flat 75 uprate 0.0075 uphold 0.999.
    Returns:
        decay factor for old = decay*old + (1-decay)*current.
    """
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    if step < flat:
        return 0.0
    decay = (step - flat) * uprate
    return min(decay, uphold)


def nft_update_old_adapter(
    peft_model: torch.nn.Module,
    global_step: int,
    decay_type: int = 2,
) -> None:
    """adaptive weight decay update old adapter: old = decay*old + (1-decay)*current"""
    decay = return_decay(global_step, decay_type)
    peft_model.set_adapter("default")
    default_params = list(
        filter(lambda p: p.requires_grad, peft_model.parameters())
    )
    peft_model.set_adapter("old")
    old_params = list(
        filter(lambda p: p.requires_grad, peft_model.parameters())
    )
    peft_model.set_adapter("default")
    with torch.no_grad():
        for src_param, tgt_param in zip(default_params, old_params):
            tgt_param.data.copy_(
                tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay)
            )