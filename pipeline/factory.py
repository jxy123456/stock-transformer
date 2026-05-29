"""组件工厂：根据 config 的 type 字段创建实例。"""

_registry = {
    "features": {},
    "model": {},
    "backtest": {},
    "stock_selector": {},
}


def register(category: str, name: str):
    """装饰器：注册组件到工厂。"""
    def deco(cls):
        _registry[category][name] = cls
        return cls
    return deco


def create_features(cfg: dict, cache, **kwargs):
    fc = cfg.get("features", {})
    name = fc.get("type", "v1_45")
    from data.features.v1_45 import V1_45FeatureEngine
    return V1_45FeatureEngine(cfg, cache, **kwargs)


def create_model(cfg: dict):
    mc = cfg.get("model", {})
    name = mc.get("type", "transformer")
    if name == "transformer":
        from model.transformer import StockMultiHorizonTransformer
        return StockMultiHorizonTransformer(
            feature_dim=cfg.get("features", {}).get("feature_dim", 45),
            seq_len=cfg.get("features", {}).get("seq_len", 120),
            d_model=mc.get("d_model", 128),
            nhead=mc.get("n_heads", 4),
            num_layers=mc.get("n_layers", 4),
            dim_feedforward=mc.get("d_ff", 256),
            dropout=mc.get("dropout", 0.20),
        )
    raise ValueError(f"Unknown model type: {name}")


def create_trainer(model, cfg: dict, device=None):
    from training.trainer import Trainer
    return Trainer(model, cfg, device=device)


def load_checkpoint(model, path: str):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
