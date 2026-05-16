# 模型训练特征清单 (80 特征)

模型输入: `(B, 250, 80)` — 250 个交易日 × 80 个特征
模型输出: 8 类收益桶概率分布

设计原则：**自身时序特征 + 横截面位置 + 市场/行业背景 + 基本面估值**

---

## A. 个股收益与趋势特征 (14) — `data/features.py`

| # | 特征 | 公式 | 说明 |
|---|------|------|------|
| 1 | log_ret_1d | log(close/close.shift(1)) | 日对数收益 |
| 2 | log_ret_5d | log(close/close.shift(5)) | 周对数收益 |
| 3 | log_ret_20d | log(close/close.shift(20)) | 月对数收益 |
| 4 | log_ret_60d | log(close/close.shift(60)) | 季对数收益 |
| 5 | log_ret_120d | log(close/close.shift(120)) | 半年对数收益 |
| 6 | intraday_ret | (close - open) / open | 日内收益率 |
| 7 | overnight_ret | (open - prev_close) / prev_close | 隔夜收益率 |
| 8 | high_low_range | (high - low) / open | 日内振幅 |
| 9 | close_position_in_bar | (close - low) / (high - low) | 收盘在 K 线中的位置 |
| 10 | close_to_ma20 | close / MA(20) - 1 | 收盘偏离 20 日均线 |
| 11 | close_to_ma60 | close / MA(60) - 1 | 收盘偏离 60 日均线 |
| 12 | close_to_ma120 | close / MA(120) - 1 | 收盘偏离 120 日均线 |
| 13 | ma20_to_ma60 | MA(20) / MA(60) - 1 | 中期趋势结构 |
| 14 | ma60_to_ma120 | MA(60) / MA(120) - 1 | 长期趋势结构 |

## B. 价格位置与回撤特征 (8) — `data/features.py`

| # | 特征 | 说明 |
|---|------|------|
| 15 | price_rank_60d | 60 日内价格分位数 |
| 16 | price_rank_250d | 250 日内价格分位数 |
| 17 | distance_to_high_60d | 距 60 日高点距离 |
| 18 | distance_to_high_250d | 距 250 日高点距离 |
| 19 | distance_to_low_60d | 距 60 日低点距离 |
| 20 | distance_to_low_250d | 距 250 日低点距离 |
| 21 | max_drawdown_250d | 250 日最大回撤 |
| 22 | rebound_from_low_60d | 距 60 日低点反弹幅度 |

## C. 成交量与流动性特征 (9) — `data/features.py` + `data/liquidity.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 23 | log_amount | features.py | 对数成交额 |
| 24 | volume_ratio_1_20 | features.py | 当日量比 |
| 25 | volume_ratio_5_20 | features.py | 5 日均量比 |
| 26 | volume_zscore_20d | features.py | 20 日量 z-score |
| 27 | turnover_ma20 | liquidity.py | 20 日均换手率 |
| 28 | turnover_ratio_5_20 | liquidity.py | 换手率短长比 |
| 29 | turnover_zscore_20d | liquidity.py | 换手率 z-score |
| 30 | amihud_illiquidity_20d | liquidity.py | Amihud 非流动性 |
| 31 | rank_log_amount_all | cross_sectional.py | 成交额全市场排名 |

## D. 波动率与风险特征 (8) — `data/features.py` + `data/market.py` + `data/cross_sectional.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 32 | realized_vol_20d | features.py | 20 日已实现波动 |
| 33 | realized_vol_60d | features.py | 60 日已实现波动 |
| 34 | downside_vol_20d | features.py | 20 日下行波动 |
| 35 | downside_vol_60d | features.py | 60 日下行波动 |
| 36 | downside_vol_ratio_20d | features.py | 下行波动占比 |
| 37 | high_low_vol_20d | features.py | 20 日高低波动 |
| 38 | rank_realized_vol_60d_all | cross_sectional.py | 全市场波动率排名 |
| 39 | market_beta_60d | market.py | 60 日 Beta |

## E. 相对市场与行业表现特征 (8) — `data/market.py` + `data/industry.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 40 | relative_ret_20d_vs_market | market.py | 20 日相对大盘超额 |
| 41 | relative_ret_60d_vs_market | market.py | 60 日相对大盘超额 |
| 42 | relative_ret_20d_vs_industry | industry.py | 20 日行业超额 |
| 43 | relative_ret_60d_vs_industry | industry.py | 60 日行业超额 |
| 44 | market_corr_60d | market.py | 60 日市场相关性 |
| 45 | industry_ret_20d | industry.py | 行业 20 日收益 |
| 46 | industry_vol_20d | industry.py | 行业 20 日波动 |
| 47 | industry_rank_ret_20d | industry.py | 行业收益在所有行业中排名 |

## F. 市场环境特征 (5) — `data/market.py` + `data/cross_sectional.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 48 | market_ret_1d | market.py | 市场日收益 |
| 49 | market_ret_20d | market.py | 市场 20 日收益 |
| 50 | market_vol_20d | market.py | 市场 20 日波动 |
| 51 | market_drawdown_60d | market.py | 市场 60 日回撤 |
| 52 | market_up_stock_ratio_1d | cross_sectional.py | 全市场上涨股票占比 |

## G. 估值特征 (7) — `data/valuation.py` + `data/cross_sectional.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 53 | earnings_yield | valuation.py | 1/PE 盈利收益率 |
| 54 | book_to_price | valuation.py | 1/PB 账面市值比 |
| 55 | sales_to_price | valuation.py | 1/PS 销售市值比 |
| 56 | pe_percentile_3y | valuation.py | PE 3 年分位数 |
| 57 | pb_percentile_3y | valuation.py | PB 3 年分位数 |
| 58 | rank_earnings_yield_industry | cross_sectional.py | 行业内 EP 排名 |
| 59 | rank_book_to_price_industry | cross_sectional.py | 行业内 BP 排名 |

## H. 基本面盈利与成长特征 (9) — `data/fundamental.py` + `data/cross_sectional.py`

| # | 特征 | 来源 | 说明 |
|---|------|------|------|
| 60 | roe_ttm | fundamental.py | 净资产收益率 |
| 61 | roa_ttm | fundamental.py | 总资产收益率 |
| 62 | gross_margin_ttm | fundamental.py | 毛利率 |
| 63 | net_margin_ttm | fundamental.py | 净利率 |
| 64 | revenue_yoy | fundamental.py | 营收同比增速 |
| 65 | net_profit_yoy | fundamental.py | 净利润同比增速 |
| 66 | revenue_yoy_acceleration | fundamental.py | 营收增速加速度 |
| 67 | profit_yoy_acceleration | fundamental.py | 利润增速加速度 |
| 68 | rank_roe_ttm_industry | cross_sectional.py | 行业内 ROE 排名 |

## I. 财报质量与资产负债特征 (8) — `data/fundamental.py`

| # | 特征 | 说明 |
|---|------|------|
| 69 | ocf_to_net_profit | 经营现金流/净利润 |
| 70 | ocf_to_revenue | 经营现金流/营收 |
| 71 | accrual_to_assets | 应计利润/总资产 |
| 72 | debt_to_asset | 资产负债率 |
| 73 | current_ratio | 流动比率 |
| 74 | equity_ratio | 权益资产比 |
| 75 | inventory_growth_minus_revenue | 存货增速-营收增速 |
| 76 | receivable_growth_minus_revenue | 应收增速-营收增速 |

## J. 横截面排名补充特征 (4) — `data/cross_sectional.py`

| # | 特征 | 说明 |
|---|------|------|
| 77 | rank_log_ret_20d_all | 20 日收益全市场排名 |
| 78 | rank_log_ret_60d_all | 60 日收益全市场排名 |
| 79 | rank_market_cap_all | 市值全市场排名 |
| 80 | rank_debt_to_asset_industry_reverse | 行业内低负债排名 |

---

## 特征来源引擎汇总

| 引擎 | 文件 | 特征数 |
|------|------|--------|
| FeatureEngine | data/features.py | 32 |
| LiquidityFeatureEngine | data/liquidity.py | 4 |
| MarketFeatureEngine | data/market.py | 9 |
| IndustryFeatureEngine | data/industry.py | 5 |
| ValuationFeatureEngine | data/valuation.py | 5 |
| FundamentalEngine | data/fundamental.py | 16 |
| CrossSectionalEngine | data/cross_sectional.py | 9 |
| **合计** | | **80** |

---

## 特征结构比例

| 层次 | 特征数 | 占比 | 含义 |
|------|--------|------|------|
| 个股自身时序 | 50 | 62.5% | A+B+C部分+D部分，Transformer 学序列演化 |
| 横截面排名 | 13 | 16.3% | C部分+D部分+G部分+H部分+J，补足市场相对位置 |
| 市场/行业背景 | 13 | 16.3% | D部分+E+F，提供背景环境 |
| 基本面/估值 | 4 | 5% | G部分+I部分，绝对值补充 |

---

## 分类目标

5 日收益桶 (8 类)：(-∞,-5%), [-5%,-2%), [-2%,-1%), [-1%,0%), [0%,1%), [1%,2%), [2%,5%), [5%,+∞)

---

## 预处理流水线

1. 特征计算 → 2. inf→NaN → 3. 热身行删除+条件特征前向填充+技术特征断言 → 4. Winsorization(5σ) → 5. Z-score(训练集fit) → 6. 最终断言无NaN/inf

---

## 防未来信息规则

1. 所有 rolling 窗口只看过去
2. 财报按 report_date(公告日) forward-fill
3. 横截面排名当日收盘后计算，次日可用
4. 标准化只在训练窗口 fit
5. 目标收益: close[t+seq_len+pred_len-1] / close[t+seq_len-1] - 1
