# Stock Transformer — A 股量化交易系统

基于 Encoder-only Transformer 的 A 股收益率预测与交易系统。输入 60 日多特征序列，预测未来 5 日收益率。

## 技术栈

Python 3.10+, PyTorch 2.0+, akshare/tushare, Parquet, YAML

## 目录职责

| 目录 | 职责 | 子文档 |
|------|------|--------|
| `data/` | 数据获取、特征工程、缓存、Dataset | [data/CLAUDE.md](data/CLAUDE.md) |
| `model/` | Transformer 架构、训练、评估 | [model/CLAUDE.md](model/CLAUDE.md) |
| `backtest/` | 事件驱动回测、A股券商、组合 | [backtest/CLAUDE.md](backtest/CLAUDE.md) |
| `risk/` | 止损/止盈/回撤熔断/仓位限制 | [risk/CLAUDE.md](risk/CLAUDE.md) |
| `strategy/` | 信号生成、仓位管理 | [strategy/CLAUDE.md](strategy/CLAUDE.md) |
| `scripts/` | CLI 入口 | [scripts/CLAUDE.md](scripts/CLAUDE.md) |
| `config/` | YAML 配置 (default.yaml + 深合并) | — |
| `utils/` | 配置加载、日志、交易日历、可视化 | — |
| `execution/` | 模拟盘、实盘接口(抽象) | — |

## 数据流

```
akshare API → fetcher → cache(Parquet) → features → preprocessor(walk-forward) → dataset
  → StockTransformer → predictions → signal_generator → risk_manager → broker → portfolio → statistics
```

## 编码规范

### 禁止

- **禁止添加任何兜底逻辑**: 不写 try-except 吞异常后返回默认值, 不写 `if x is None: x = default`, 不写 `except Exception: pass`, 不写 `or` 兜底 (如 `data or []`)。数据缺失/配置错误/操作失败 → 直接抛异常崩溃, 暴露问题而非掩盖。
- **禁止 look-ahead bias**: 标准化只在训练窗口 fit, 信号 T 收盘 → T+1 执行, 财报 forward-fill 按发布时间。
- **禁止当日收盘价成交**: 订单必须在次日开盘价执行。
- **禁止绕过 T+1**: 卖出必须经 `portfolio.todays_buys` 检查。
- **禁止忽略 A 股规则**: 涨跌停/印花税/最低佣金/100 股手数, 缺一不可。
- **禁止硬编码超参**: 所有可调参数必须在 `config/default.yaml` 中定义。
- **禁止注释说代码做了什么**: 只在 WHY 不显然时写一行注释。

### 修改时必须验证

- 改 `data/preprocessor.py` → 标准化参数不泄露未来信息
- 改 `backtest/broker.py` → A 股规则 (T+1/涨跌停/印花税) 仍正确
- 改 `backtest/engine.py` → 信号-执行时序 (T → T+1) 仍正确
- 新增特征 → 在 `FeatureEngine.feature_columns` 或 `FundamentalEngine.feature_columns` 注册
- 新增配置项 → 同步更新 `config/default.yaml`
