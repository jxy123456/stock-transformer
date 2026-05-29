# Stock Data — A 股量化实验

基于 akshare/tushare 双源的 A 股数据、特征、训练和回测实验项目。

## 技术栈

Python 3.10+, akshare/tushare, pandas, Parquet, PyYAML, loguru

## 目录职责

| 目录 | 职责 |
|------|------|
| `data/` | 数据获取、股票池、特征工程 |
| `scripts/` | CLI 入口 (download_data / update_data) |
| `config/` | YAML 配置、实验配置、股票池 |
| `pipeline/` | 数据集生成、实验编排 |
| `training/` | Dataset、训练、评估 |
| `backtest/` | 回测引擎 |
| `utils/` | 配置加载、日志 |

## 数据流

```
akshare/tushare API → cache(Parquet) → features(39列) → datasets(.npy) → training/backtest
```

## 特征口径

- 当前模型输入为 39 个特征，配置中 `feature_dim` 应为 39。
- 行业收益、行业波动、行业超额收益、行业内排名已移除；当前不依赖行业指数或行业收益序列。
- 特征缓存版本见 `data.features.base.FEATURE_CACHE_VERSION`。版本变化后必须重建 `outputs/features` 和 `outputs/datasets`。
- 财务数据按公告日后的下一个交易日生效，避免公告日收盘后信息进入当日信号。

## 标签口径

- 当前模型只训练 5d 和 20d 两个预测头，不再训练 1d 预测头。
- 5d/20d 标签不是第 5 天或第 20 天单点收益，而是未来 5/20 个交易日平均收盘价相对当前收盘价的收益。
- 回测分数由 5d/20d 期望收益加权得到，默认权重为 `{5d: 0.6, 20d: 0.4}`。

## 训练资源

- 默认训练配置启用 `training.data_parallel: true` 和 `training.mixed_precision: true`。
- 单机多卡训练使用 PyTorch `DataParallel`，无需改变启动命令。
- `training.batch_size` 是全局 batch size，会被拆分到多张 GPU；显存仍有余量时优先增大该值。
- `training.num_workers` 和 `training.prefetch_factor` 控制 DataLoader 预取，GPU 利用率低时先检查 CPU/data loading 是否成为瓶颈。

## 编码规范

### 禁止

- **禁止硬编码超参**: 所有可调参数必须在 `config/default.yaml` 中定义。
- **禁止注释说代码做了什么**: 只在 WHY 不显然时写一行注释。

### 修改时必须验证

- 改 `data/fetcher.py` → 列名标准化仍正确
- 改 `data/cache.py` → Parquet 读写仍正确
- 新增配置项 → 同步更新 `config/default.yaml`
