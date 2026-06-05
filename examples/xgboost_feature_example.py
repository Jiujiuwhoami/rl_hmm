"""
XGBoost 特征工程示例

演示如何使用 rl_hmm 的 FeatureEngineer 进行特征提取
"""

import pandas as pd
import numpy as np
from rl_hmm import Config, FeatureEngineer, FeatureRegistry


def example_feature_engineering():
    """
    基础特征工程示例
    """
    print("=" * 60)
    print("特征工程示例")
    print("=" * 60)

    # 创建模拟数据
    n = 1000
    np.random.seed(42)
    
    # 模拟 OHLCV 数据
    price = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({
        'Open': price + np.random.randn(n),
        'High': price + 0.5 + np.abs(np.random.randn(n)),
        'Low': price - 0.5 - np.abs(np.random.randn(n)),
        'Close': price,
        'Volume': np.random.randint(1000, 10000, n)
    })

    # 配置和添加特征
    config = Config()
    config.features = [
        'return', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'volatility', 'atr', 'sma_20', 'ema_20', 'bollinger_upper',
        'bollinger_lower', 'vol_change', 'sma_60', 'vol_ma_ratio'
    ]

    df_features = FeatureEngineer.add_features(df, config.features)
    
    print(f"\n原始数据列: {list(df.columns)}")
    print(f"添加特征后列: {list(df_features.columns)}")
    print(f"\n前 5 行数据:")
    print(df_features[['Close', 'return', 'rsi', 'macd', 'atr']].head())
    print("\n" + "=" * 60)


def example_custom_feature_registration():
    """
    动态注册自定义特征示例
    """
    print("\n" + "=" * 60)
    print("自定义特征注册示例")
    print("=" * 60)

    # 动态注册新特征
    @FeatureRegistry.register('price_to_sma_ratio')
    def price_to_sma_ratio(df):
        """收盘价与20日均线的比值"""
        sma_20 = df['Close'].rolling(20).mean()
        return df['Close'] / sma_20

    @FeatureRegistry.register('momentum')
    def momentum(df, period=10):
        """动量指标"""
        return df['Close'].pct_change(period)

    @FeatureRegistry.register('price_range')
    def price_range(df):
        """价格波动范围"""
        return (df['High'] - df['Low']) / df['Close']

    # 查看所有可用特征
    all_features = FeatureRegistry.get_available_features()
    print(f"\n所有可用特征: {all_features}")

    # 使用自定义特征
    config = Config()
    config.features = [
        'return', 'rsi', 'price_to_sma_ratio', 'momentum', 'price_range'
    ]

    # 模拟数据并测试
    n = 200
    df = pd.DataFrame({
        'Open': np.random.randn(n) + 100,
        'High': np.random.randn(n) + 102,
        'Low': np.random.randn(n) + 98,
        'Close': np.random.randn(n) + 100,
        'Volume': np.random.randint(1000, 10000, n)
    })

    df_features = FeatureEngineer.add_features(df, config.features)
    print(f"\n使用的特征: {config.features}")
    print(f"自定义特征已添加: {[c for c in df_features.columns if c in config.features]}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_feature_engineering()
    example_custom_feature_registration()
