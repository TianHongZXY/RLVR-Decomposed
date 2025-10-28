<div align="center">

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.01347)
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/tianhongzxy/status/1929596099154633036)
[![Hugging Face](https://img.shields.io/badge/RLVR_Decomposed-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/TianHongZXY/rlvr-decomposed-683c0cd7151b769d8ea5915c)

</div>

## News
- **[2025/10/27]** An updated version is out! It includes additional experiments and results (Llama model, λ ablation, and more). Check it out [here](https://arxiv.org/pdf/2506.01347)! ✨
- **[2025/09/18]** Our paper is accepted to NeurIPS 2025! 🎉
- **[2025/06/01]** We release our [paper](https://arxiv.org/pdf/2506.01347) and code. 🚀
## Quick Start
### Installation
Our code is implemented based on [verl](https://github.com/volcengine/verl). We recommend to use docker image provided by verl, please refer to their [documents](https://verl.readthedocs.io/en/v0.2.x/start/install.html).

Start from a custom environment:
```
conda create -y -n verl python=3.10.14 && conda activate verl
pip install -e .
pip install vllm==0.8.2
pip install latex2sympy2
pip install fire
pip install tensordict==0.7.2
python -m pip install flash-attn --no-build-isolation
```
**For training Qwen3 models, you will need to upgrade to `vllm==0.8.5` and `transformers==4.52.2`.**

## Training
PSR, NSR, W-REINFORCE: specify `advantage` in `run_qwen2.5-math-7b_psr_nsr.sh` to train the model with PSR, NSR, or W-REINFORCE. For W-REINFORCE, set `positive_advantage_weight`, which corresponds to the λ in the paper, recommended value is 0.1.
```
bash run_qwen2.5-math-7b_psr_nsr.sh
```
PPO
```
bash run_qwen2.5-math-7b_ppo.sh
```
GRPO
```
bash run_qwen2.5-math-7b_grpo.sh
```

## Evaluation
Specify `MODEL_PATH` and `OUTPUT_DIR` in `eval.sh`, then `bash eval.sh`.

Calculate Pass@k: `python calculate_metrics --file_path <file_to_evaluate>`

## Troubleshoot
- Out-of-Memory error: Decrease `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`, `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`, `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`

- Frozen after `Started a local Ray instance.`: Add `num_cpus=N` to `ray.init()` in `verl/trainer/main_ppo.py`,  for example, `ray.init(num_cpus=4, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})`

 ## Citation

If you find our paper or code useful, please consider cite our work:

```bibtex
@article{zhu2025rlvr-decomposed,
  title={The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning},
  author={Zhu, Xinyu and Xia, Mengzhou and Wei, Zhepei and Chen, Wei-Lin and Chen, Danqi and Meng, Yu},
  journal={arXiv preprint arXiv:2506.01347},
  year={2025}
}
```
