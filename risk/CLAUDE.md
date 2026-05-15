# risk/ — 风控模块

## 模块职责

| 文件 | 职责 |
|------|------|
| `manager.py` | 风控协调器, 组合三个子模块的过滤逻辑 |
| `stop_loss.py` | 固定止损/止盈/追踪止损 |
| `position_limit.py` | 持仓数量 + 单只仓位上限 |
| `drawdown_control.py` | 回撤熔断 + 渐进减仓 |

## RiskManager.filter() 流程

```
输入: signals + portfolio + current_prices

1. 检查现有持仓的止损/止盈/追踪止损 → 生成强制卖出信号
2. 遍历新信号:
   a. 如果是买入 + 回撤熔断激活 → 跳过
   b. 检查仓位限制 (数量/集中度) → 不通过则跳过
   c. 如果是买入 + 回撤超 10% → 按缩减因子降低 strength
3. 返回: 强制卖出 + 过滤后的新信号
```

## 止损/止盈 (stop_loss.py)

| 规则 | 默认值 | 触发条件 |
|------|--------|----------|
| 固定止损 | -8% | (current_price - avg_cost) / avg_cost ≤ -0.08 |
| 止盈 | +15% | (current_price - avg_cost) / avg_cost ≥ 0.15 |
| 追踪止损 | 5% | current_price < highest_price × 0.95 |

- `highest_price` 在每次 `check()` 中更新
- 三个条件按顺序检查, 命中任一即生成卖出信号, 不再检查后续

## 仓位限制 (position_limit.py)

| 规则 | 默认值 |
|------|--------|
| 最多持仓 | 5 只 |
| 单只上限 | 组合净值 25% |

## 回撤控制 (drawdown_control.py)

| 阶段 | 回撤阈值 | 行为 |
|------|----------|------|
| 正常 | < 10% | 不干预 |
| 减仓 | 10% ~ 15% | 新买入 strength × (1 - (dd - 10%) / 5%) |
| 熔断 | ≥ 15% | 禁止所有新买入 |

- 回撤 = (peak_value - current_value) / peak_value
- peak_value 取权益曲线的历史最高

## 关键约束

- 止损/追踪止损在每交易日收盘后检查, 产生的卖出信号在次日开盘执行
- 回撤控制只限制**新买入**, 不强制卖出已有持仓
- `RiskManager.on_market_close()` 必须调用以更新回撤状态
