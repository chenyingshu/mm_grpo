# Async Training Strategies
GeRL provides a general framework for generative model RL post-training. By default, it uses a hybrid engine where actor, rollout, reference, and other components share the same resources.

## Supported Strategies
- [x] [Decoupled actor and rollout](#decoupled-actor-and-rollout), support asynchronous rollout.
- [x] [Rollout with async reward computing](#rollout-with-async-reward-computing): asynchronous batch reward computing during rollout generation
- [x] [One-step-off async policy](#one-step-off-async-policy): one-step-off asynchronous policy training with decoupled trainer and rollout.
<!-- - [ ] fully-async policy: fully asynchronous off-policy training with decoupled trainer and rollout. -->

## Decoupled Actor and Rollout
> [!NOTE]
> **NOT used by default. Coupled actor and rollout is used by default.**

**Introduction:** Decouple actor and rollout into standalone resource pools, and use async rollout.

## Usage
`hybrid_engine` is applied by default, i.e., coupled actor and rollout.<br/>
To decouple actor and rollout into standalone resource pools, and use async rollout, set configs:
```bash
actor_rollout_ref.hybrid_engine=False
actor_rollout_ref.rollout.mode="async"
```

## Rollout with Async Reward Computing
> [!NOTE]
> **Used by default.**

**Introduction:** During the rollout generation loop, after generating responses for each micro batch, asynchronously launch reward computation for the current batch. By combining asynchronous reward computing with rollout generation, rollout's GPU idle time is significantly reduced.

<div align="center">
<img width="1024" alt="with_reward" src="https://github.com/user-attachments/assets/430c3604-4af8-4e01-aabf-3ce2ac0648b0" />
<br/>
Left: Synchronous reward computing. Right: Asynchronous reward computing during rollout.
</div>
<br/>

Reference: [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch/blob/main/scripts/train.py#L355).

## Usage
The `with_reward` function is applied by default by setting following config:
```bash
actor_rollout_ref.rollout.with_reward=True
```

## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

The following table shows the training throughput increase when using asynchronous reward computing (with_reward=True) compared to synchronous reward computing (with_reward=False):

| Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | `with_reward` | Batch Size | `rollout.n` | training samples per step| `ppo_micro_batch_size_per_gpu` | Speedup (sec/step)| Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO-Fast | True | 1 | paddle-ocr | True | 8 | 16 | 8x16=128 | 8 | 21| +5% |
| SD3.5-M | Flow-GRPO-Fast | True | 1 | qwenvl-ocr-vllm | True |  8 | 16 | 8x16=128 | 8 | 19| +5% |
| SD3.5-M | Flow-GRPO-Fast | True | 8 | paddle-ocr | True |  64 | 16 |64x16=1024 | 8 | 150| +100%|


## One-Step-Off Async Policy
> [!NOTE]
> **Used by default when using decoupled actor and rollout.**

**Introduction:**
We support the one-step-off async trainer to parallelize the generation and training processes, utilizing samples generated in the previous step for the current training.
It involves appropriately partitioning resources, allocating dedicated resources for rollout generation and actor training. By reducing resources allocated to the generation phase, GPU idle time during long-tail sample generation is mitigated. Throughout this process, generation and training parameters maintain a one-step off policy.

<div align="center">
<img src="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/docs/one_step_off_policy.png" width=1024 />
<br/>
Left: Synchronous training. Right: One-step-off asynchronous training
</div>
<br/>

References:
[verl Recipe: One Step Off Policy Async Trainer](https://github.com/volcengine/verl/tree/main/recipe/one_step_off_policy); <br/>
[Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models](https://arxiv.org/abs/2410.18252)

## Usage
To apply `one-step-off` async strategy, set configs:
```bash
actor_rollout_ref.hybrid_engine=False
actor_rollout_ref.rollout.mode="async"
actor_rollout_ref.async_strategy="one-step-off"
```


## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

- Training GPU hours required to reach and maintain a validation reward score of approximately 0.8:


| Model   | Algorithm | Hybrid Engine | # Cards |   Reward Fn  | Async Strategy | # GPUs for Actor | # GPUs for Rollout |  Batch Size | `rollout.n` | Learning Rate | # Val Samples | Throughput | # GPU Hour | Script |
| ------- | ------- | ------- | ------  | --------- | --------- | --------- | --------- | ---------- | ------------- | ---------- | ---------- | ------ |  ------ | ------ |
| SD3.5-M | Flow-GRPO-Fast | False |  3  | qwenvl-ocr-vllm* | one-step-off    | 1 | 2 | 8       | 8 |  1e-4          | 32 |  1.07        | 0.68 |[run_sd3_fast_3p_a1_r2.sh](./run_sd3_fast_3p_a1_r2.sh) |
| SD3.5-M | Flow-GRPO-Fast | False |  3  | qwenvl-ocr-vllm* | one-step-off    | 2 | 1 |  16       | 8 |  1e-4          | 32 |  1.25        | 1.13| [run_sd3_fast_3p_a2_r1.sh](./run_sd3_fast_3p_a2_r1.sh) |
| SD3.5-M | Flow-GRPO-Fast     | True | 3  | qwenvl-ocr-vllm*  | -   | 3 | 3 | 24      | 8 |  1e-4          | 33 |    1.42      | 1.02 | - |


**\*Note**: `UnifiedReward-Think-qwen3vl-32b` model was used in reward computing.

- Validation reward curve：

<div align="center">
<img width="600" alt="3p_comparison" src="https://github.com/user-attachments/assets/a9630a75-5cbf-48fe-996c-6c66a0b5f8be" />
</div>
