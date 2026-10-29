# 32B 继续预训练（单机 8×RTX 6000D）

基于 FSDP2 的继续预训练（CPT）脚本，面向一台 8 卡 PCIe 机器（无 NVLink）。依赖只有 torch、transformers 和 numpy。读取 parquet 语料时还需要 pyarrow。

| 文件 | 作用 |
|---|---|
| `pack_data.py` | 把 jsonl/parquet/txt 语料 tokenize，每篇文档末尾加 EOS，拼接写入 `<out>.bin`（uint32） |
| `train_fsdp.py` | FSDP2 训练：bf16 计算、全量激活重计算、可选 CPU offload、按权重混合多份语料、断点自动续训 |
| `launch.sh` | torchrun 单机 8 卡启动脚本 |
| `export_hf.py` | 把检查点转回 HF 格式（bf16 safetensors） |

## 步骤

```bash
# 0. 开训前在机器上确认（只读）
nvidia-smi topo -m; free -g; df -h /nfs
python -c "import torch;print([[torch.cuda.can_device_access_peer(i,j) for j in range(8)] for i in range(8)])"

# 1. 打包语料：领域语料和通用语料（replay）分开打包，训练时按权重混合
python cpt/pack_data.py --tokenizer $MODEL --inputs /data/domain  --out /data/packed/domain
python cpt/pack_data.py --tokenizer $MODEL --inputs /data/general --out /data/packed/general

# 2. 冒烟测试：跑 5 步，确认显存、吞吐、loss 正常，检查点能存下来
MODEL=$MODEL DATA="/data/packed/domain:0.8 /data/packed/general:0.2" OUT=/nfs/ckpt/smoke \
  MAX_STEPS=5 SAVE_EVERY=5 bash cpt/launch.sh

# 3. 正式训练；中断后用同样的命令重跑，会从 OUT 下最新的检查点继续
MODEL=$MODEL DATA="/data/packed/domain:0.8 /data/packed/general:0.2" OUT=/nfs/ckpt/cpt-32b \
  TRAIN_TOKENS=1e9 bash cpt/launch.sh

# 4. 导出
python cpt/export_hf.py --base $MODEL --ckpt /nfs/ckpt/cpt-32b/step-0000477 --out /nfs/models/cpt-32b
```

## 默认配置及原因

- **纯数据并行 + FSDP2 分片**：PCIe 上做张量并行或流水线并行很慢，或者显存太紧。
- **`--cpu-offload`（launch.sh 默认开启）**：32B 全参的权重、梯度和 Adam 状态约 512 GB，8×85 GB 不 offload 放不下激活。开启后这些都放在 CPU 内存，约 400 GB 以上，机器有 1.5 TB；优化器在 CPU 上更新，会慢 20% 到 30%。如果冒烟测试发现显存有富余，可以设 `EXTRA_ARGS=""` 关掉 offload 试试。
- **每卡每个 micro-step 处理 16k token（4096×4）**，全局 batch 约 2M token。每个 micro-step 都做一次梯度 reduce-scatter；用 `--reduce-dtype bf16` 可以减半 PCIe 流量。
- **学习率 2e-5**（约为原峰值的 1/10），1% warmup 后 cosine 衰减到 10%。
- **语料混合**：领域语料 70% 到 90%，通用语料 replay 10% 到 30%，防止通用能力退化。CPT 之后还需要再做 SFT。
- **检查点**：DCP 分片格式，含优化器状态，32B 约 400 GB 以上，必须放在 NFS 上，不能放在 Pod 的临时盘。默认只保留最近 2 个。

## 时长估算（2026-10-29 在 RTX 6000D 上实测 bf16 matmul 约 142 TFLOPS）

按有效利用率 40% 到 50% 估算，8 卡约 2,300 到 3,000 tok/s，每天约 2 亿 token：1B token 约 4 到 5 天，5B 约 3 周，10B 约 6 到 7 周。日志里每一步都会打印 tok/s、MFU 和剩余时间（ETA），冒烟测试后以实测为准。

## 已验证与未验证

- 已在 CPU（gloo，2 进程，tiny Qwen2）上验证：权重加载正确（第一步 loss 与 HF 原模型一致）、loss 下降、断点续训后 loss 逐步完全一致、导出 HF 后 rotary 等 buffer 正确。
- **未验证**：GPU 上的 `--cpu-offload` 路径，以及 32B 的实际显存和吞吐。需要在 8 卡机器上跑一次冒烟测试确认。
