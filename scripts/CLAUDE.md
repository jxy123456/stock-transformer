# scripts/ — CLI 入口

## 脚本清单

### download_data.py — 全量下载

```bash
# 下载全 A 股 (约 5000 只) 的行情 + 财报
python scripts/download_data.py --all --start 20150101

# 下载指定股票
python scripts/download_data.py --symbols 000001 600519 --start 20150101

# 只下行情, 跳过财报
python scripts/download_data.py --all --no-fundamental

# 跳过已有缓存的股票 (断点续传)
python scripts/download_data.py --all --skip-cached

# 查看缓存
python scripts/download_data.py --list-cache
```

流程: 行情(OHLCV) → 估值(PE/PB) → 财报(摘要) → 特征计算 → CSV

`--all` 先调 `ak.stock_zh_a_spot_em()` 拉全 A 股列表, 过滤掉北交所(8xxxxx)和新三板(4xxxxx), 只保留沪深主板(60/00) + 创业板(300) + 科创板(688)。

失败的股票代码写入 `outputs/logs/failed_{type}.txt`, 可据此重跑。

### update_data.py — 增量更新

```bash
# 增量更新全 A 股
python scripts/update_data.py --all

# 更新指定股票
python scripts/update_data.py --symbols 000001 600519

# 只更新行情
python scripts/update_data.py --all --no-fundamental

# 强制全量重下
python scripts/update_data.py --all --force
```

增量逻辑: 读本地 Parquet 最后日期 → 从 last_date+1 拉到今天 → append_daily() 追加去重 → 重算特征

### train.py — Walk-Forward 训练

```bash
python scripts/train.py --symbol 000001 --start 20200101 --end 20241231
python scripts/train.py --symbol 000001 --device cuda
```

流程: 加载数据 → 计算特征 → 生成 walk-forward folds → 每 fold 训练新模型 → 评估测试集 → 输出汇总指标

### backtest.py — 回测

```bash
python scripts/backtest.py --symbols 000001 600519 --start 20220101 --end 20241231
python scripts/backtest.py --symbols 000001 --cash 500000
```

输出: `outputs/backtest_results/` 下 `backtest_report.txt` + `equity_curve.csv` + `trades.csv`

### paper_trade.py — 模拟盘

```bash
python scripts/paper_trade.py --symbols 000001 --checkpoint outputs/checkpoints/model_best.pt
```

加载训练好的模型, 获取当日最新数据, 推理 → 信号 → 执行 → 记录

### analyze.py — 分析结果

```bash
python scripts/analyze.py
```

读取 backtest_results, 生成权益曲线图 + 交易汇总

## 关键约束

- download 和 update 在请求间 sleep 0.3~0.5s, 避免被封
- `--skip-cached` 用于断点续传: 中断后重跑不会重复下载已有数据
- update 增量更新时, 财报/估值直接全量覆盖 (数据量小), 行情增量追加
- train 脚本中 target 是日收益率: (close[t] - close[t-1]) / close[t-1]
- backtest 脚本在无模型时使用简单动量信号作为 fallback
