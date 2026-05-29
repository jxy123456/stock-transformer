# Stock Data — A 股数据下载

基于 akshare/tushare 双源的 A 股历史数据下载工具。

## 技术栈

Python 3.10+, akshare/tushare, pandas, Parquet, PyYAML, loguru

## 目录职责

| 目录 | 职责 |
|------|------|
| `data/` | 数据获取 (fetcher) + 本地缓存 (cache) |
| `scripts/` | CLI 入口 (download_data / update_data) |
| `config/` | YAML 配置 (default.yaml) |
| `utils/` | 配置加载、日志、交易日历 |

## 数据流

```
akshare/tushare API → fetcher → cache(Parquet) → outputs/data_cache/
```

## 编码规范

### 禁止

- **禁止硬编码超参**: 所有可调参数必须在 `config/default.yaml` 中定义。
- **禁止注释说代码做了什么**: 只在 WHY 不显然时写一行注释。

### 修改时必须验证

- 改 `data/fetcher.py` → 列名标准化仍正确
- 改 `data/cache.py` → Parquet 读写仍正确
- 新增配置项 → 同步更新 `config/default.yaml`
