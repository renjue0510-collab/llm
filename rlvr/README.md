# RLVR 基线（单机 8×RTX 6000D）

论文选题「用策略模型隐状态做策略级探索，防止 RLVR 的 pass@k 坍缩」的第一步：先在一台 8 卡机器上跑通 GRPO/DAPO 基线，并画出 base 模型的 pass@k 曲线。框架是 verl 0.9 + vLLM。选题调研见 [Claude Doc](https://claude.ai/artifact/4aRKUmdLoizhiWZMmTBicw)。

| 文件 | 作用 |
|---|---|
| `check_env.sh` | 只读检查：GPU 占用、torch / vLLM / verl 等版本 |
| `prepare_data.py` | 生成 verl 格式 parquet：DAPO-Math-17k 去重后 17,398 题做训练；MATH-500、AIME24、AIME25、AMC23 做评测；`val.parquet` = AIME24+AIME25+AMC23（100 题）用于训练中验证 |
| `reward.py` | 规则奖励：取最后一个 `\boxed{}`，用 math_verify 判等价，对 1 错 0。训练和评测共用 |
| `run_grpo.sh` | verl GRPO 训练：组大小 16、每步 128 题、无 KL、clip-higher 0.28、token-mean |
| `passk_eval.py` | vLLM 每卡一个进程采样，算 pass@1…pass@256（无偏估计） |
| `test_cpu.py` | 奖励和 pass@k 估计的 CPU 单测 |

## 步骤

```bash
# 0. 环境（只读）。verl 0.9 需要 vllm>=0.18、transformers 5.5.3–5.10
bash rlvr/check_env.sh
pip install verl==0.9.0 math-verify   # 缺什么装什么，别动已有的 torch / vllm

# 1. 数据（需要能访问 huggingface.co，或设置 HF_ENDPOINT 镜像）
python rlvr/prepare_data.py --out /nfs/data/rlvr

# 2. 冒烟测试：3 步、每步 16 题 × 8 条、回复上限 1024，确认能跑通并存下检查点
SMOKE=1 MODEL=/nfs/models/Qwen3-1.7B-Base DATA=/nfs/data/rlvr OUT=/nfs/ckpt/rlvr/smoke bash rlvr/run_grpo.sh

# 3. 正式基线（每个种子一台机器，5 台机器可同时跑不同种子或不同方法）
SEED=1 MODEL=/nfs/models/Qwen3-1.7B-Base DATA=/nfs/data/rlvr OUT=/nfs/ckpt/rlvr/grpo-1.7b-s1 bash rlvr/run_grpo.sh

# 4. base 模型的 pass@k 曲线（训练前就可以跑）
python rlvr/passk_eval.py --model /nfs/models/Qwen3-1.7B-Base --data /nfs/data/rlvr/math500.parquet \
    --n 256 --out /nfs/eval/base-1.7b/math500 --save-responses

# 5. 训练后的检查点：先转 HF，再跑同样的评测
python -m verl.model_merger merge --backend fsdp \
    --local_dir /nfs/ckpt/rlvr/grpo-1.7b-s1/global_step_500/actor --target_dir /nfs/models/grpo-1.7b-s1
```

## 默认配置及原因

- **DAPO 风格的 GRPO**：不加 KL、clip 上界 0.28、token 级平均。这是目前 RLVR 论文最常用的强基线，后续方法都在它上面改。
- **vLLM TP=1**：8B 以下模型单卡放得下，PCIe 上做张量并行反而慢。
- **训练提示词统一为 `\boxed{}` 格式**：DAPO 原始数据用 `Answer:` 格式，这里改写成和评测一致，保证训练奖励和评测指标是同一个判分函数。
- **训练中验证**用 AIME24/25 + AMC23 共 100 题，每题采 8 次（temperature 0.6），日志里看 mean@8。完整的 pass@k 曲线用 `passk_eval.py` 单独跑。
- **每步 128 题 × 16 条 = 2,048 条 rollout**，回复上限 4096 token。吞吐还没实测，冒烟测试后按日志里的每步耗时调 `BATCH` 和 `STEPS`。

## 已验证与未验证

- **已验证（CPU）**：数据真实下载并生成（训练 17,398 题，评测 500/30/30/40 题），所有评测题的标准答案自身都能判对；`test_cpu.py` 4 项通过；`run_grpo.sh` 的全部 50 个参数已用 verl 0.9.0 的 Hydra 配置实际组装，没有无效键。
- **未验证**：GPU 上的训练和 vLLM 采样都还没跑过，需要在 8 卡机器上跑一次冒烟测试。
