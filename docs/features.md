# 特征说明

当前实验使用 `data.features.v1_45.V1_45FeatureEngine` 生成模型输入。类名保留 `v1_45` 是为了兼容已有配置，实际输入特征数为 39。

## 当前特征

- 价格类：`open_gap`、`intraday_ret`、`close_ret_1d`、`amplitude`、`high_to_preclose`、`low_to_preclose`、`close_to_high`
- 收益类：`ret_5d`、`ret_20d`、`ret_60d`
- 波动和回撤：`volatility_20d`、`volatility_60d`、`max_drawdown_20d`
- 均线偏离：`close_to_ma20`、`close_to_ma60`
- 价格位置：`price_position_20d`、`price_position_60d`
- 成交量和成交额：`volume_chg_5d`、`amount_chg_5d`、`volume_ratio_20d`、`amount_ratio_20d`
- 换手率：`turnover_rate`、`turnover_to_20d`
- 市值：`log_total_market_cap_max_norm`、`log_float_market_cap_max_norm`、`total_market_cap_rank_market`
- 估值：`earnings_yield`、`book_to_price`
- 基本面：`revenue_yoy`、`net_profit_yoy`、`gross_margin`、`net_margin`、`roe`、`debt_to_asset`、`ocf_to_net_profit`
- 大盘：`market_ret_20d`、`market_volatility_20d`、`excess_ret_market_20d`
- 截面排名：`ret_20d_rank_market`

## 已移除的行业特征

行业相关输入已移除，包括：

- `industry_ret_20d`
- `industry_volatility_20d`
- `excess_ret_industry_20d`
- `total_market_cap_rank_industry`
- `pe_rank_industry`
- `ret_20d_rank_industry`

移除原因：当前缓存只有行业映射，没有可靠的行业收益序列；行业内排名在 `liquid100` 股票池中覆盖也不稳定，容易产生整列缺失或大量补零。

## 缓存要求

特征缓存版本由 `data.features.base.FEATURE_CACHE_VERSION` 控制。行业特征移除后版本为 `v3_no_industry`。重新训练或回测前需要先运行数据管线重建：

```bash
python pipeline/data_pipeline.py
```
