# data/ — 数据管线

## 模块职责

| 文件 | 职责 |
|------|------|
| `fetcher.py` | 双源数据获取 (akshare 主 / tushare 备), 列名标准化 |
| `cache.py` | Parquet 本地缓存, 增量追加, 按日期去重 |
| `features.py` | 技术指标计算, 纯 numpy/pandas 实现 |
| `fundamental.py` | 基本面特征 (PE/PB/ROE), 季频 → 日频 forward-fill |
| `preprocessor.py` | Walk-forward fold 生成 + 标准化 (zscore/minmax) |
| `dataset.py` | PyTorch Dataset, 滑动窗口切分 (seq_len → pred_len) |

## 数据源 API

| 数据 | 方法 | akshare API | 频率 |
|------|------|-------------|------|
| 日线 OHLCV | `fetch_daily()` | `stock_zh_a_hist` | 日频 |
| 指数行情 | `fetch_index()` | `stock_zh_index_daily` | 日频 |
| 估值 PE/PB | `fetch_valuation_metrics()` | `stock_a_indicator_lg` | 日频 |
| 利润表 | `fetch_income_statement()` | `stock_financial_report_sina` | 季频 |
| 资产负债表 | `fetch_balance_sheet()` | 同上 | 季频 |
| 现金流量表 | `fetch_cashflow_statement()` | 同上 | 季频 |
| 财报摘要 | `fetch_financial_summary()` | `stock_financial_abstract_ths` | 季频 |

行情数据使用前复权 (`adjust="qfq"`)。

## 缓存

- 每个 symbol 一个 Parquet: `daily_{symbol}.parquet`, `valuation_{symbol}.parquet`, `financial_{symbol}.parquet`
- 增量: `cache.append_daily()` 追加 + 按 datetime 去重
- `cache.get_last_date()` 返回最后日期供增量脚本使用

## 技术特征清单

`FeatureEngine.compute()` 在 OHLCV DataFrame 上计算:

- MA: SMA(5/10/20/60) + close/MA 比率
- EMA(12/26)
- MACD: DIF/DEA/柱状图 (12/26/9)
- RSI(14)
- 布林带: 上轨/下轨/位置百分比 (20日, 2σ)
- ATR(14)
- 量比: volume/MA(volume,20), log1p(volume)
- 收益率: 1日/5日/20日
- 波动率: 5日/20日滚动 std
- 振幅, 开盘价与前收比

所有列名在 `FeatureEngine.feature_columns` 中注册。

## 基本面特征清单

`FundamentalEngine.compute_features()` 在估值 DataFrame 上计算:

- PE/PB 原始值 + 250 日分位数
- ROE 及其变化
- 营收增速及加速度
- 净利增速及加速度
- 股息率

季频财报 forward-fill 到日频: 财报发布后沿用至下一季报发布。**这保证了不引入未来信息。**

## Walk-Forward 验证

`Preprocessor.generate_folds()` 滑动窗口:

- 训练 500 天, 验证 100 天, 测试 50 天, 步长 50 天
- `prepare_fold_data()`: 切片 → 在训练窗口 fit 标准化 → transform 验证/测试
- **关键**: 标准化参数只在训练窗口计算, 绝不泄露未来信息

## 关键约束

- `fetcher._standardize()` 统一中文/英文列名为 `datetime/open/high/low/close/volume/amount/turnover`
- `TimeSeriesDataset` 的 `__getitem__` 返回 (input_seq, target_seq), target 是未来 pred_len 步的收益率
- 所有技术指标都是向后看的 (rolling/ewm), 不存在 look-ahead
