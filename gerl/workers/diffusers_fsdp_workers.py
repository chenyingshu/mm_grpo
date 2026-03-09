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
The main entry point to run the FlowGRPO algorithm
"""

import copy
import datetime
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Optional

import psutil
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    Execute,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)

from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import (
    reduce_timing,
    topk_reduce_ratio_min_max,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config.optimizer import build_optimizer
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy

from ..protocol import DataProto
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.nft import nft_update_old_adapter
from ..utils.lora import collect_lora_params_for_adapter, select_lora_modules
from .config import (
    DiffusersModelConfig,
    DiffusionRolloutConfig,
    FSDPEngineConfig,
)
from .rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class DiffusersActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(
                    seconds=self.config.get("nccl_timeout", 600)
                ),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )

        # create training dispatch
        self._register_dispatch_collect_info(
            "actor", dp_rank=self.rank, is_collect=True
        )

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None
            or self._lora_rank > 0
        )

        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get(
            "use_orig_params", False
        )
        self._use_diffusion_nft = (
            OmegaConf.select(
                self.config, "actor.policy_loss.loss_mode", default="flow_grpo"
            )
            == "diffusion_nft"
        )

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(
            omega_profiler_config, dataclass_type=ProfilerConfig
        )
        if omega_profiler_config.get("tool", None) in [
            "npu",
            "nsys",
            "torch",
            "torch_memory",
        ]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(
                    omega_profiler_config.get("tool")
                )
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank, config=profiler_config, tool_config=tool_config
            ),
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size()
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            assert (
                self.config.actor.ppo_mini_batch_size
                % self.config.actor.ppo_micro_batch_size_per_gpu
                == 0
            ), (
                f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
            )
            assert (
                self.config.actor.ppo_mini_batch_size
                // self.config.actor.ppo_micro_batch_size_per_gpu
                > 0
            ), (
                f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
            )

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from diffusers.models.transformers import SD3Transformer2DModel
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from Diffusers", logger=logger)
        local_path = model_path

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=False, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # TODO (Mike): generalize to other diffusers model later
            actor_module = SD3Transformer2DModel.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                subfolder="transformer",
            )

            actor_module.requires_grad_(not self._is_lora)

            if use_fused_kernels:
                actor_module.fuse_qkv_projections()

            if enable_gradient_checkpointing:
                actor_module.enable_gradient_checkpointing()

            if self._is_lora:
                print("Applying LoRA to actor module")

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    print(
                        f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}"
                    )

                    # Copy adapter to local if needed
                    local_adapter_path = copy_to_local(
                        lora_adapter_path,
                        use_shm=self.config.model.get("use_shm", False),
                    )

                    actor_module.load_lora_adapter(local_adapter_path)
                else:
                    from peft import LoraConfig

                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "init_lora_weights": self.config.model.lora_init_weights,
                        "target_modules": convert_to_regular_types(
                            select_lora_modules(
                                model_name=os.path.basename(model_path),
                                target_modules=self.config.model.target_modules,
                            )
                        ),
                        "exclude_modules": convert_to_regular_types(
                            self.config.model.exclude_modules
                        ),
                        "bias": "none",
                    }
                    actor_module.add_adapter(LoraConfig(**lora_config))

                # DiffusionNFT: add "old" adapter for rollout policy (dual adapter)
                if self._use_diffusion_nft and role == "actor" and self._is_actor:
                    from peft import LoraConfig

                    lora_config_for_old = actor_module.peft_config["default"]
                    if isinstance(lora_config_for_old, dict):
                        lora_config_for_old = LoraConfig(**lora_config_for_old)
                    actor_module.add_adapter(lora_config_for_old, "old")
                    # Initial copy: default -> old
                    actor_module.set_adapter("default")
                    default_params = list(
                        filter(lambda p: p.requires_grad, actor_module.parameters())
                    )
                    actor_module.set_adapter("old")
                    old_params = list(
                        filter(lambda p: p.requires_grad, actor_module.parameters())
                    )
                    actor_module.set_adapter("default")
                    with torch.no_grad():
                        for src_param, tgt_param in zip(default_params, old_params):
                            tgt_param.data.copy_(src_param.detach().data)
                    if self.rank == 0:
                        logger.info(
                            "DiffusionNFT: added 'old' LoRA adapter, copied default->old"
                        )

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(
            f"After init {role} from Diffusers Pipeline", logger=logger
        )

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = (
                    None if role == "actor" else CPUOffloadPolicy(pin_memory=True)
                )

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(
                    fsdp_size=self.device_mesh.shape[-1]
                ),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(
                actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing
            )
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        if role == "actor" and self.config.model.use_ema:
            from ..utils.ema import EMAModuleWrapper

            ema_wrapper = EMAModuleWrapper(
                parameters=actor_module_fsdp.parameters(),
                decay=self.config.model.ema_decay,
                device=get_device_name(),
            )
        else:
            ema_wrapper = None

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import (
                get_constant_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
            )

            actor_optimizer = build_optimizer(
                actor_module_fsdp.parameters(), optim_config
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(
                    f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}"
                )

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(
                    f"LR scheduler type {lr_scheduler_type} is not supported"
                )

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, ema_wrapper

    def _build_scheduler(self, model_path):
        # TODO (Mike): generalize to other diffusers scheduler later
        from .diffusers_model.schedulers import FlowMatchSDEDiscreteScheduler

        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_path, subfolder="scheduler"
        )
        return scheduler

    def _build_rollout(self):
        # 1. parse rollout and huggingface model config
        rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(
            self.config.rollout
        )
        model_config: DiffusersModelConfig = omega_conf_to_dataclass(self.config.model)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = (
            self.config.rollout.tensor_model_parallel_size
            * self.config.rollout.data_parallel_size
        )
        infer_world_size = infer_tp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(dp, infer_tp),
            mesh_dim_names=["dp", "infer_tp"],
        )

        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout",
            dp_rank=rollout_device_mesh["dp"].get_local_rank(),
            is_collect=is_collect,
        )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(
            gen_dp_rank + 1000
        )  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(
            f"Before building {self.config.rollout.name} rollout", logger=logger
        )
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage(
            f"After building {self.config.rollout.name} rollout", logger=logger
        )

        # used for LoRA
        self.base_sync_done = True
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self, swap_ema: bool = False):
        """Context switch hybridengine to rollout mode."""
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        if swap_ema:
            assert self.ema_wrapper is not None
            self.ema_wrapper.copy_ema_to_model(self.actor_module_fsdp.parameters())

        peft_config = None
        peft_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        use_old_adapter = (
            self._use_diffusion_nft
            and hasattr(peft_model, "peft_config")
            and "old" in getattr(peft_model, "peft_config", {})
        )
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            if use_old_adapter:
                params = collect_lora_params_for_adapter(
                    module=self.actor_module_fsdp,
                    adapter_name="old",
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=self.base_sync_done,
                )
            else:
                params = collect_lora_params_for_adapter(
                    module=self.actor_module_fsdp,
                    adapter_name="default",
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=self.base_sync_done,
                )
        else:
            params = self.actor_module_fsdp.state_dict()

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if swap_ema:
            per_tensor_param = copy.deepcopy(list(per_tensor_param))
            self.ema_wrapper.copy_temp_to_model(self.actor_module_fsdp.parameters())

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume()
        log_gpu_memory_usage("After resume weights", logger=logger)
        await self.rollout.update_weights(
            per_tensor_param,
            peft_config=peft_config,
            base_sync_done=self.base_sync_done,
        )
        log_gpu_memory_usage("After update_weights", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .actor import DiffusersPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )

        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor:
            optim_config = self.config.actor.optim
            fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.ema_wrapper,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                role="actor",
                enable_activation_offload=self.config.model.get(
                    "enable_activation_offload", False
                ),
            )

            self.scheduler = self._build_scheduler(model_path=local_path)

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor model during init", logger=logger
                )

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DiffusersPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                scheduler=self.scheduler,
                actor_optimizer=self.actor_optimizer,
                ema_wrapper=self.ema_wrapper,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_fused_kernels=use_fused_kernels,
                role="ref",
            )[0]
            self.scheduler = self._build_scheduler(model_path=local_path)
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DiffusersPPOActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
                scheduler=self.scheduler,
            )

        if self._is_actor:
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                checkpoint_config=self.config.actor.checkpoint,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=get_device_id()
            )

        data = data.to(
            "cpu"
        )  # data will to device with each micro batch on actor.update_policy

        metrics = self.actor.update_policy(data=data)
        metrics["perf/max_memory_allocated_gb"] = (
            get_torch_device().max_memory_allocated() / (1024**3)
        )
        metrics["perf/max_memory_reserved_gb"] = (
            get_torch_device().max_memory_reserved() / (1024**3)
        )
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
        self.actor_lr_scheduler.step()

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})

        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during update_actor", logger=logger
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor", logger=logger
            )

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_old_adapter(self, global_step: int):
        """DiffusionNFT: adapative weight decay update old adapter after policy update."""
        if not self._is_actor or not self._use_diffusion_nft:
            return
        peft_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        if not (
            hasattr(peft_model, "peft_config")
            and "old" in getattr(peft_model, "peft_config", {})
        ):
            return
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        decay_type = OmegaConf.select(
            self.config, "actor.policy_loss.nft_decay_type", default=2
        )
        nft_update_old_adapter(peft_model, global_step, decay_type=int(decay_type))
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        timing_generate: dict[str, float] = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            swap_ema = self.config.model.use_ema and prompts.meta_info.get(
                "validate", False
            )
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode(swap_ema=swap_ema))
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = (
            topk_reduce_ratio_min_max(timing_generate["generate_sequences"])
        )
        timing_generate = reduce_timing(timing_generate)
        timing_reward = output.meta_info.pop("timing_reward", None)
        if timing_reward is not None:
            timing_reward = reduce_timing(timing_reward)
            timing_generate.update(timing_reward)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares

        is_lora = data.meta_info.pop("is_lora", False)
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size_per_gpu
        )
        # perform recompute log_prob
        if is_lora:
            self.actor.actor_module.disable_adapters()
        old_log_probs, prev_sample_mean = self.actor.compute_log_prob(data=data)
        if is_lora:
            self.actor.actor_module.enable_adapters()
        output = DataProto.from_dict(
            tensors={
                "old_log_probs": old_log_probs,
                "prev_sample_mean": prev_sample_mean,
            }
        )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during compute_log_prob", logger=logger
            )

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this prev_sample_mean is in fact ref_prev_sample_mean
            data = DataProto.from_dict(
                tensors={"ref_prev_sample_mean": data.batch["prev_sample_mean"]}
            )
            return data
        assert self._is_ref
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size

        data = data.to(
            "cpu"
        )  # data will to device with each micro batch on ref.compute_log_prob
        _, output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"ref_prev_sample_mean": output})

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, global_step=0, max_ckpt_to_keep=None):
        # TODO (Mike): need to save the EMA model as well

        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        torch.distributed.barrier()

        if self._is_lora and hasattr(
            getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"
        ):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            lora_save_path_old = os.path.join(local_path, "lora_adapter_old")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            peft_config_old = {}
            if torch.distributed.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                os.makedirs(lora_save_path_old, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
                if "old" in peft_model.peft_config:
                    peft_config_old = asdict(peft_model.peft_config.get("old", {}))
                    peft_config_old["peft_type"] = peft_config_old["peft_type"].value
                    peft_config_old["target_modules"] = list(
                        peft_config_old["target_modules"]
                    )
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(
                        get_device_name()
                    )
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    lora_params_old = None
                    if "old" in getattr(peft_model, "peft_config", {}):
                        lora_params_old = collect_lora_params_for_adapter(
                            module=self.actor_module_fsdp,
                            adapter_name="old",
                            layered_summon=False,
                            base_sync_done=True,
                        )
                    if torch.distributed.get_rank() == 0:
                        save_file(
                            lora_params,
                            os.path.join(lora_save_path, "adapter_model.safetensors"),
                        )
                        with open(
                            os.path.join(lora_save_path, "adapter_config.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
                        if lora_params_old is not None:
                            save_file(
                                lora_params_old,
                                os.path.join(
                                    lora_save_path_old, "adapter_model.safetensors"
                                ),
                            )
                            with open(
                                os.path.join(lora_save_path_old, "adapter_config.json"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                json.dump(
                                    peft_config_old, f, ensure_ascii=False, indent=4
                                )
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})",
                    rank=torch.distributed.get_rank(),
                    logger=logger,
                    log_only_rank_0=True,
                )

            torch.distributed.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=torch.distributed.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            del_local_after_load=del_local_after_load,
        )

        # DiffusionNFT dual-adapter usage, ensure shape remains valid after resume.
        if self._use_diffusion_nft and self._is_lora:
            peft_model = getattr(
                self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
            )
            if hasattr(peft_model, "peft_config") and "old" not in peft_model.peft_config:
                from peft import LoraConfig

                lora_config_for_old = peft_model.peft_config.get("default")
                if isinstance(lora_config_for_old, dict):
                    lora_config_for_old = LoraConfig(**lora_config_for_old)
                peft_model.add_adapter(lora_config_for_old, "old")
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
                        tgt_param.data.copy_(src_param.detach().data)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(
        self, tag: str = "manual", sub_dir: Optional[str] = None
    ) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = (
                        OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    )
                    self.profiler._impl.sampler.dump_memory_snapshot(
                        out_dir=out_dir, tag=tag, sub_dir=sub_dir
                    )
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class AsyncDiffusersActorRolloutRefWorker(DiffusersActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_params(self, swap_ema: bool = False):
        if self.ema_wrapper is None:
            swap_ema = False

        if swap_ema:
            self.ema_wrapper.copy_ema_to_model(self.actor_module_fsdp.parameters())

        base_sync_done = getattr(self, "base_sync_done", True)
        peft_config = None
        peft_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        use_old_adapter = (
            self._use_diffusion_nft
            and hasattr(peft_model, "peft_config")
            and "old" in getattr(peft_model, "peft_config", {})
        )
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            if use_old_adapter:
                params = collect_lora_params_for_adapter(
                    module=self.actor_module_fsdp,
                    adapter_name="old",
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=base_sync_done,
                )
            else:
                params = collect_lora_params_for_adapter(
                    module=self.actor_module_fsdp,
                    adapter_name="default",
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=base_sync_done,
                )
        else:
            params = self.actor_module_fsdp.state_dict()

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if swap_ema:
            per_tensor_param = copy.deepcopy(list(per_tensor_param))
            self.ema_wrapper.copy_temp_to_model(self.actor_module_fsdp.parameters())

        return {"params": per_tensor_param, "config": peft_config}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def update_weights(self, params_with_config: dict):
        per_tensor_param, peft_config = (
            params_with_config["params"],
            params_with_config["config"],
        )
        await self.rollout.update_weights(
            per_tensor_param,
            peft_config=peft_config,
            base_sync_done=self.base_sync_done,
        )

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"),
        blocking=False,
    )
    def generate_sequences(self, prompts):
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        timing_generate: dict[str, float] = {}

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = (
            topk_reduce_ratio_min_max(timing_generate["generate_sequences"])
        )
        timing_generate = reduce_timing(timing_generate)
        timing_reward = output.meta_info.pop("timing_reward", None)
        if timing_reward is not None:
            timing_reward = reduce_timing(timing_reward)
            timing_generate.update(timing_reward)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        return output
