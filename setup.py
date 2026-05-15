from setuptools import find_packages, setup

setup(
    name="stock_transformer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "akshare>=1.12.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "loguru>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "stock-fetch=scripts.fetch_data:main",
            "stock-train=scripts.train:main",
            "stock-backtest=scripts.backtest:main",
            "stock-analyze=scripts.analyze:main",
        ],
    },
)
