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

import json
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_reference_policy,
    need_reward_model,
)
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
    should_save_ckpt_esi,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from ...protocol import DataProto
from ...utils.tracking import ValidationGenerationsLogger
from ..config.algorithm import AlgoConfig
from .async_utils import update_weights
from .core_algos import (
    AdvantageEstimator,
    compute_flow_grpo_outcome_advantage,
)
from .metric_utils import (
    compute_diffusion_data_metrics,
    compute_diffusion_throughout_metrics,
    compute_diffusion_timing_metrics,
)
from .reward import CPUAsyncRewardWorker, compute_reward


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        global_std (bool, optional): Whether to use global standard deviation for advantage normalization.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    if adv_estimator == AdvantageEstimator.FLOW_GRPO:
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_flow_grpo_outcome_advantage(
            instance_level_rewards=data.batch["instance_level_rewards"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            global_std=global_std,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayDiffusionPPOTrainer:
    """
    Trainer for diffusion-based GRPO using Ray.
    """

    def __init__(
        self,
        config,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        self.diffusion_nft = (
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode == "diffusion_nft"
        )

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )
        else:
            assert Role.Actor in role_worker_mapping, f"{role_worker_mapping.keys()=}"
            assert Role.Rollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(
        self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]
    ):
        """
        Creates the train and validation dataloaders.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get(
                "gen_batch_size", self.config.data.train_batch_size
            ),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps
                    )
        except Exception as e:
            print(
                f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}"
            )

    def _dump_generations(
        self,
        inputs: list[str],
        outputs: torch.FloatTensor,
        gts: list[Optional[str]],
        scores: list[float],
        reward_extra_infos_dict: dict[str, list],
        dump_path: str,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)

        visual_folder = os.path.join(dump_path, f"{self.global_steps}")
        os.makedirs(visual_folder, exist_ok=True)

        output_paths = []
        images_pil = outputs.cpu().float().permute(0, 2, 3, 1).numpy()
        images_pil = (images_pil * 255).round().clip(0, 255).astype("uint8")
        for i, image in enumerate(images_pil):
            image_path = os.path.join(visual_folder, f"{i}.jpg")
            Image.fromarray(image).save(image_path)
            output_paths.append(image_path)

        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data: dict[str, list] = {
            "input": inputs,
            "output": output_paths,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            scores = batch.batch["instance_level_scores"].cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in batch
            ]

            self._dump_generations(
                inputs=batch.non_tensor_batch["prompt"].tolist(),
                outputs=batch.batch["responses"],
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"})

        # pop those keys for generation
        batch_keys_to_pop: list[str] = []
        non_tensor_batch_keys_to_pop = (
            set(batch.non_tensor_batch.keys()) - reward_model_keys
        )
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop or async reward compute during rollout, we need reward model keys to compute score.
        if (
            self.async_rollout_manager
            or self.config.actor_rollout_ref.rollout.with_reward
        ):
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _gen_next_batch(self, gen_batch, timing_raw, reward_fn=None):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        swap_ema = gen_batch.meta_info.get("validate", False)
        with marked_timer("update_weight", timing_raw, color="purple"):
            # sync weights from actor to rollout if actor and rollout do not share resource pool
            update_weights(self.actor_rollout_wg, self.rollout_wg, swap_ema=swap_ema)

        # apply async reward during rollout
        if self.config.actor_rollout_ref.rollout.with_reward:
            gen_batch.meta_info["reward_fn"] = reward_fn

        # sync or async rollout generation
        gen_batch_output = self.rollout_wg.generate_sequences(gen_batch)

        return gen_batch_output

    def _validate(self):
        if self.val_reward_fn is None:
            raise ValueError("val_reward_fn must be provided for validation.")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [
                        str(uuid.uuid4())
                        for _ in range(len(test_batch.non_tensor_batch["prompt"]))
                    ],
                    dtype=object,
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            sample_inputs.extend(test_batch.non_tensor_batch["prompt"])
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            test_gen_batch.meta_info = {
                "noise_level": self.config.actor_rollout_ref.rollout.val_kwargs.noise_level,
                "num_inference_steps": self.config.actor_rollout_ref.rollout.val_kwargs.num_inference_steps,
                "seed": self.config.actor_rollout_ref.rollout.val_kwargs.seed,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            if not self.async_rollout_manager:
                test_output_gen_batch = self._gen_next_batch(
                    test_gen_batch, {}, reward_fn=self.compute_val_reward_async
                )
                if (
                    self.async_rollout_mode
                ):  # do not run async rollout during validation
                    test_output_gen_batch = test_output_gen_batch.get()
            else:
                test_output_gen_batch = self.async_rollout_manager.generate_sequences(
                    test_gen_batch
                )

            print("validation generation end")

            # Store generated outputs
            output_images = test_output_gen_batch.batch["responses"]
            sample_outputs.append(output_images)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            reward_extra_info = None
            if self.config.actor_rollout_ref.rollout.with_reward:
                reward_tensor = test_output_gen_batch.batch["instance_level_scores"]
                reward_extra_info = test_output_gen_batch.non_tensor_batch
            else:
                # evaluate using reward_function
                reward_tensor, reward_extra_info = compute_reward(
                    test_batch, self.val_reward_fn
                )

            scores = reward_tensor.tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if reward_extra_info:
                for key, lst in reward_extra_info.items():
                    if key != "prompt":
                        reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        sample_outputs = torch.cat(sample_outputs, dim=0)
        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [
                        int(name.split("@")[-1].split("/")[0])
                        for name in metric2val.keys()
                    ]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "maj", "best"]
                        )
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            actor_role = (
                Role.ActorRolloutRef
                if Role.ActorRolloutRef in self.role_worker_mapping
                else Role.ActorRollout
            )
        else:
            actor_role = Role.Actor

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = (
                actor_rollout_cls
            )
        else:
            for role in [Role.Actor, Role.Rollout]:
                resource_pool = self.resource_pool_manager.get_resource_pool(role)
                role_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[role],
                    config=self.config.actor_rollout_ref,
                    role=str(role),
                )
                self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = (
                ref_policy_cls
            )

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
            self.rm_wg = None
            self.compute_reward_async = None
            self.compute_val_reward_async = None
        elif (
            self.config.actor_rollout_ref.rollout.with_reward
            or self.config.reward_model.launch_reward_fn_async
        ):
            # use a lightweight CPU reward worker for async reward compute
            self.rm_wg = CPUAsyncRewardWorker.remote(
                self.config, reward_fn=self.reward_fn
            )
            self.val_rm_wg = CPUAsyncRewardWorker.remote(
                self.config, reward_fn=self.val_reward_fn
            )
            self.compute_reward_async = self.rm_wg.compute_reward
            self.compute_val_reward_async = self.val_rm_wg.compute_reward
        else:
            self.rm_wg = None
            self.compute_reward_async = None
            self.compute_val_reward_async = None

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if (
            OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout")
            is not None
        ):
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(
                self.config.global_profiler, "steps"
            )
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                    is not None
                ), (
                    "worker_nsight_options must be set when using nsys with profile_steps"
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if not self.hybrid_engine:
            self.rollout_wg = all_wg[str(Role.Rollout)]
            self.rollout_wg.init_model()
        else:
            self.rollout_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.one_step_off_policy = False  # naive synchronous training
        self.async_rollout_manager = None
        if self.config.actor_rollout_ref.rollout.mode == "async":
            # Async mode currently does not require a scheduler because requests are handled directly by the worker group.
            # If future async implementations require scheduling, create a follow-up issue to track this work.
            self.async_rollout_mode = True

            if not self.hybrid_engine and (
                self.config.actor_rollout_ref.async_strategy == "one-step-off"
            ):
                self.one_step_off_policy = True

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), (
                    "resume ckpt must be str type"
                )
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(
                role="e2e", profile_step=self.global_steps
            )
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def fit(self):
        """
        The training loop of diffusion GRPO.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        is_first_step = True
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [
                        str(uuid.uuid4())
                        for _ in range(len(batch.non_tensor_batch["prompt"]))
                    ],
                    dtype=object,
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                # one-step-off async policy, start first async generation
                if is_first_step and self.one_step_off_policy:
                    # Start the first asynchronous generation task.
                    batch_data_future = self._gen_next_batch(
                        gen_batch, timing_raw, self.compute_reward_async
                    )
                    is_first_step = False
                    continue

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Sampling Part

                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.one_step_off_policy:
                            # get previous generation
                            gen_batch_output = batch_data_future.get()

                            # update weights and async next generation
                            if not is_last_step:
                                batch_data_future = self._gen_next_batch(
                                    gen_batch, timing_raw, self.compute_reward_async
                                )
                        elif not self.async_rollout_manager:
                            gen_batch_output = self._gen_next_batch(
                                gen_batch, {}, self.compute_reward_async
                            )
                            # Currently, non-one-step-off async policy does not really run async rollout.
                            if self.async_rollout_mode:
                                gen_batch_output = gen_batch_output.get()
                        else:
                            gen_batch_output = (
                                self.async_rollout_manager.generate_sequences(gen_batch)
                            )

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # Reward Part
                    if not self.config.actor_rollout_ref.rollout.with_reward:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                raise NotImplementedError  # TODO： reward model worker

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = self.compute_reward_async.remote(
                                    data=batch
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(
                                    batch, self.reward_fn
                                )

                    # Training Part
                    rollout_corr_config = self.config.algorithm.get(
                        "rollout_correction", None
                    )
                    bypass_recomputing_logprobs = (
                        rollout_corr_config
                        and rollout_corr_config.get("bypass_mode", False)
                    )
                    if self.diffusion_nft:
                        # for NFT policy loss
                        if bypass_recomputing_logprobs:
                            batch.batch["old_preds"] = batch.batch["rollout_preds"]
                        else:  # Recompute preds
                            with marked_timer("old_preds", timing_raw, color="blue"):
                                old_preds = self.actor_rollout_wg.compute_pred(batch)
                                batch = batch.union(old_preds)

                    else:  # apply flow-grpo by default
                        if bypass_recomputing_logprobs:
                            batch.batch["old_log_probs"] = batch.batch[
                                "rollout_log_probs"
                            ]
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(
                                    batch
                                )
                                batch = batch.union(old_log_prob)

                        assert "old_log_probs" in batch.batch, (
                            f'"old_log_prob" not in {batch.batch.keys()=}'
                        )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(
                            str(Role.RefPolicy), timing_raw, color="olive"
                        ):
                            # TODO: apply diffusionNFT
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                            else:
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                )
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if not self.config.actor_rollout_ref.rollout.with_reward:
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]

                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(
                                    future_reward
                                )
                            batch.batch["instance_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {
                                        k: np.array(v)
                                        for k, v in reward_extra_infos_dict.items()
                                    }
                                )

                            batch.batch["instance_level_rewards"] = batch.batch[
                                "instance_level_scores"
                            ]

                        # TODO: apply diffusionNFT
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            global_std=self.config.algorithm.global_std,
                            config=self.config.algorithm,
                        )

                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
                        )

                # Validation Part

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.test_freq == 0
                    )
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print(
                            "Force saving checkpoint: ESI instance expiration approaching."
                        )
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_diffusion_data_metrics(batch=batch))
                metrics.update(
                    compute_diffusion_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_diffusion_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool
                    == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
