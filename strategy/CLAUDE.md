# strategy/ — 交易策略

## 模块职责

| 文件 | 职责 |
|------|------|
| `signal_generator.py` | 模型预测 → 买入/卖出信号 |
| `position_sizer.py` | 信号 → 具体股数 (100 股整数倍) |
| `ensemble.py` | 多模型加权集成预测 |

## 信号生成 (signal_generator.py)

`SignalGenerator.generate_signals(predictions, current_positions, current_prices)`:

| 条件 | 动作 |
|------|------|
| pred_return > 0.5% 且未持仓 | 买入, strength = pred_return / 0.5% (上限 2.0) |
| pred_return < -0.5% 且已持仓 | 卖出, strength = \|pred_return / -0.5%\| (上限 2.0) |
| 其他 | 不操作 |

- 阈值在 `config/strategy.buy_threshold` / `sell_threshold` 中配置
- 已持仓的股票不重复买入
- strength 用于仓位管理, 值越大仓位越重

## 仓位管理 (position_sizer.py)

`PositionSizer.size(signal, portfolio_value, price, volatility)` → 股数

| 方法 | 逻辑 | 配置键 |
|------|------|--------|
| `vol_adjusted` | 目标 2% 波动率贡献, fraction = 0.02 / vol, 上限 max_position_pct | `strategy.sizing_method` |
| `fixed_fraction` | portfolio_value × max_position_pct × strength | 同上 |
| `kelly` | portfolio_value × max_position_pct × strength (简化版) | 同上 |

结果向下取整到 100 股整数倍, 最少 100 股。

## 集成 (ensemble.py)

`EnsembleStrategy.predict(x)` = Σ(weight_i × model_i(x))

权重归一化, 默认等权。

## 关键约束

- 信号基于**预测收益率**, 不是预测价格
- 仓位计算结果必须 ≥ 100 股, 否则不交易
- 波动率估计值有下限 0.5%, 防止除零
