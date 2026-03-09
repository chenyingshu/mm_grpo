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

"""LoRA parameter collection utilities for old-adapter support."""

from collections import OrderedDict

from peft.utils.save_and_load import get_peft_model_state_dict

from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import FSDP, fsdp_version


def collect_lora_params_for_adapter(
    module,
    adapter_name: str,
    layered_summon: bool,
    base_sync_done: bool,
) -> OrderedDict:
    """
    Collect LoRA params for a specific adapter (e.g. "old" for DiffusionNFT rollout).
    Mirrors verl's collect_lora_params but passes adapter_name to get_peft_model_state_dict.
    When adapter_name != "default", layered_summon is not supported (uses non-layered path).
    """
    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)

    if fsdp_version(module) > 0:
        if layered_summon and adapter_name == "default":
            from verl.utils.fsdp_utils import layered_summon_lora_params

            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
        else:
            # Non-layered path (or adapter_name != "default" - layered_summon doesn't support adapter_name)
            with FSDP.summon_full_params(module, writeback=False):
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(
                        peft_model, adapter_name=adapter_name
                    )
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = (
                        "cpu"
                        if "cpu" in str(next(model.parameters()).device)
                        else get_device_name()
                    )
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace(
                            "_fsdp_wrapped_module.", ""
                        ).replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
                    from verl.utils.device import get_torch_device

                    get_torch_device().empty_cache()
    else:
        if base_sync_done:
            lora_params = get_peft_model_state_dict(
                peft_model, adapter_name=adapter_name
            )
        else:
            model = peft_model.base_model.model
            orig_dev = (
                "cpu"
                if "cpu" in str(next(model.parameters()).device)
                else get_device_name()
            )
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(
                    ".base_layer", ""
                )
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)

    return lora_params