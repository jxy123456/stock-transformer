# backtest/ — 事件驱动回测引擎

## 模块职责

| 文件 | 职责 |
|------|------|
| `engine.py` | 回测主循环, 协调数据→策略→风控→执行→组合 |
| `events.py` | 事件类型: MarketEvent/SignalEvent/OrderEvent/FillEvent |
| `broker.py` | A 股模拟券商: T+1/涨跌停/印花税/佣金/滑点/手数 |
| `portfolio.py` | 组合状态: 持仓/现金/权益曲线/T+1 追踪 |
| `statistics.py` | 绩效指标: Sharpe/Sortino/回撤/Calmar/胜率/盈亏比 |

## A 股交易规则 (broker.py)

全部在 `ASHareBroker.execute_order()` 中实现, 返回 `FillEvent` 或 `None`(拒绝):

### 1. T+1 交收
- 当日买入的股票不可当日卖出
- 实现: `portfolio.todays_buys` 集合追踪当日买入, 卖出时检查
- `Portfolio.on_market_close()` 清空 `todays_buys`

### 2. 涨跌停
- 主板 (60xxxx/00xxxx): ±10%
- 创业板 (300xxxx): ±20%
- 科创板 (688xxxx): ±20%
- 超限价订单直接拒绝 (返回 None)

### 3. 印花税
- 0.1%, **仅卖出**时收取
- 加在佣金之上

### 4. 佣金
- 0.03% 双边
- 最低 5 元/笔

### 5. 滑点
- 5 个基点
- 买入: open × (1 + 5bps)
- 卖出: open × (1 - 5bps)

### 6. 手数
- 100 股整数倍
- 不足 100 股拒绝

### 7. 执行时序 (engine.py)
- **T 日收盘**: 观察数据, 生成信号
- **T+1 日开盘**: 执行订单
- 实现: `pending_orders` 缓冲, 当日信号次日执行

## 回测主循环 (engine.py)

```
for each trading day:
  1. 执行昨日 pending_orders (T+1 开盘价)
  2. 生成新信号 (T 收盘)
  3. 风控过滤
  4. 仓位计算 → 转为 OrderEvent
  5. 存入 pending_orders (明日执行)
  6. Portfolio.on_market_close() → 更新权益曲线
```

## 绩效指标 (statistics.py)

`BacktestStatistics.compute()` 返回:
- `total_return`, `annualized_return`
- `sharpe_ratio`, `sortino_ratio`
- `max_drawdown`, `calmar_ratio`
- `win_rate`, `profit_factor`
- `n_trades`, `total_commission`, `final_value`

## 关键约束

- **永远不要在 T 日执行 T 日产生的信号** — 这是回测最常见的作弊
- `Portfolio.on_fill()` 买入时检查现金是否足够, 不足则调整数量
- `Portfolio.on_market_close()` 必须调用以清空 `todays_buys` 和记录权益曲线
- 涨跌停判断基于 `prev_close`, 不是当日开盘价
