from setuptools import find_packages, setup

setup(
    name="stock_data",
    version="0.3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "akshare>=1.12.0",
        "tushare>=1.4.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "stock-download=scripts.download_data:main",
            "stock-update=scripts.update_data:main",
        ],
    },
)
