# model/ — Transformer 模型

## 模块职责

| 文件 | 职责 |
|------|------|
| `transformer.py` | StockTransformer 主模型 + EncoderBlock |
| `attention.py` | MultiHeadSelfAttention + AttentionPooling |
| `feed_forward.py` | PositionWiseFeedForward (GELU) |
| `positional_encoding.py` | 正弦/可学习位置编码 |
| `trainer.py` | 训练循环, 早停, LR 调度, 梯度裁剪 |
| `evaluator.py` | 评估指标: 方向准确率/IC/Rank IC/MSE |
| `checkpoint.py` | 模型保存/加载/自动清理 |

## 架构

```
Input (batch, 60, num_features)
  → Linear(num_features, 128)
  → PositionalEncoding(128)
  → 4× EncoderBlock (Pre-LN)
      LayerNorm → MultiHeadAttention(8 heads, d_k=16) → residual + dropout
      LayerNorm → FFN(128→512→128, GELU) → residual + dropout
  → LayerNorm
  → AttentionPooling(128)
  → Linear(128→64) → GELU → Dropout → Linear(64→5)
  → Output (batch, 5)  — 未来5天预测收益率
```

设计决策:
- **Encoder-only**: 无自回归误差累积, 回归任务足够
- **Pre-LN**: LayerNorm 在 attention/FFN 之前, 训练更稳定
- **AttentionPooling**: 学习式注意力池化, 让模型聚焦最相关的时间步
- **输出是收益率而非价格**: 收益率平稳, 易学习, 直接用于信号

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| loss | Huber (delta=1.0) | 对涨跌停极端值鲁棒 |
| optimizer | AdamW | lr=5e-4, weight_decay=0.01 |
| scheduler | CosineAnnealingWarmRestarts | T_0=10, T_mult=2 |
| warmup | 5 epochs | 线性增长到 lr |
| gradient_clip | max_norm=1.0 | — |
| early_stop | patience=15 | 验证集 loss |

## 评估指标

`Evaluator.compute_metrics()` 返回:
- `directional_accuracy`: sign(pred) == sign(actual) 的比例, 交易中最重要
- `ic`: Pearson 相关系数 (预测 vs 实际收益率)
- `rank_ic`: Spearman 秩相关系数
- `mse`, `mae`, `rmse`

## 检查点

- 命名: `{symbol}_fold{N}_{timestamp}.pt`, 最佳模型: `{symbol}_fold{N}_best.pt`
- 保存内容: model_state, optimizer_state, epoch, val_loss
- 自动清理: 保留 best 文件 + 最近 3 个

## 关键约束

- `trainer.train()` 每个 fold 创建全新模型, 不跨 fold 复用
- `trainer.predict()` 在 eval 模式下运行, 无 dropout
- Xavier uniform 初始化所有 dim>1 的参数
- Huber Loss 的 delta=1.0 意味着误差超过 1% 的部分按线性处理
