import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional


class FeatureRegistry:
    """特征注册器 - 用于动态注册和管理特征函数"""
    
    # 特征注册表：{特征名称: (特征函数, 是否启用)}
    _features: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """装饰器：注册特征函数"""
        def decorator(func: Callable) -> Callable:
            cls._features[name] = func
            return func
        return decorator
    
    @classmethod
    def get_all_features(cls) -> List[str]:
        """获取所有已注册的特征名称"""
        return list(cls._features.keys())
    
    @classmethod
    def get_feature_func(cls, name: str) -> Optional[Callable]:
        """获取指定特征的函数"""
        return cls._features.get(name)
    
    @classmethod
    def clear(cls):
        """清空所有注册的特征"""
        cls._features.clear()


# ============ 内置特征实现 ============

@FeatureRegistry.register('return')
def feature_return(df: pd.DataFrame) -> pd.Series:
    """对数收益率"""
    return np.log(df['Close'] / df['Close'].shift(1))


@FeatureRegistry.register('macd')
def feature_macd(df: pd.DataFrame) -> pd.Series:
    """MACD 指标"""
    close = df['Close'].values
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    return pd.Series(ema_fast - ema_slow)


@FeatureRegistry.register('macd_signal')
def feature_macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD 信号线"""
    close = df['Close'].values
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    return pd.Series(macd).ewm(span=9, adjust=False).mean()


@FeatureRegistry.register('macd_hist')
def feature_macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD 柱状图"""
    close = df['Close'].values
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    return pd.Series(macd - macd_signal)


@FeatureRegistry.register('rsi')
def feature_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """RSI 相对强弱指标"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@FeatureRegistry.register('atr')
def feature_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR 平均真实波动"""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0
    return pd.Series(tr).rolling(window=length).mean()


@FeatureRegistry.register('volatility')
def feature_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """滚动波动率"""
    if 'return' in df.columns:
        return df['return'].rolling(window).std()
    return df['Close'].pct_change().rolling(window).std()


@FeatureRegistry.register('vol_change')
def feature_vol_change(df: pd.DataFrame) -> pd.Series:
    """成交量变化率"""
    return df['Volume'].pct_change()


@FeatureRegistry.register('vol_ma_ratio')
def feature_vol_ma_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """成交量与均线比率"""
    return df['Volume'] / df['Volume'].rolling(window).mean()


@FeatureRegistry.register('sma_20')
def feature_sma_20(df: pd.DataFrame) -> pd.Series:
    """20日简单移动平均"""
    return df['Close'].rolling(20).mean()


@FeatureRegistry.register('sma_60')
def feature_sma_60(df: pd.DataFrame) -> pd.Series:
    """60日简单移动平均"""
    return df['Close'].rolling(60).mean()


@FeatureRegistry.register('ema_20')
def feature_ema_20(df: pd.DataFrame) -> pd.Series:
    """20日指数移动平均"""
    return df['Close'].ewm(span=20, adjust=False).mean()


@FeatureRegistry.register('bollinger_upper')
def feature_bollinger_upper(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """布林带上轨"""
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return sma + 2 * std


@FeatureRegistry.register('bollinger_lower')
def feature_bollinger_lower(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """布林带下轨"""
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return sma - 2 * std


class FeatureEngineer:
    """技术指标特征提取器 - 支持动态特征配置"""
    
    # 默认启用的特征
    DEFAULT_FEATURES = [
        'return',
        'macd_hist',
        'rsi',
        'atr',
        'vol_change',
        'volatility',
        'vol_ma_ratio'
    ]
    
    @staticmethod
    def add_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        添加指定的特征到DataFrame
        
        Args:
            df: 原始数据DataFrame
            features: 要添加的特征名称列表，None表示使用默认特征
            
        Returns:
            添加特征后的DataFrame
        """
        df = df.copy()
        
        # 如果未指定特征列表，使用默认特征
        feature_list = features if features is not None else FeatureEngineer.DEFAULT_FEATURES
        
        # 依次添加每个特征
        for feature_name in feature_list:
            func = FeatureRegistry.get_feature_func(feature_name)
            if func is not None:
                df[feature_name] = func(df)
            else:
                raise ValueError(f"未知特征: {feature_name}")
        
        df.fillna(0, inplace=True)
        return df
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加所有已注册的特征"""
        return FeatureEngineer.add_features(df, FeatureRegistry.get_all_features())
    
    @staticmethod
    def get_available_features() -> List[str]:
        """获取所有可用的特征名称"""
        return FeatureRegistry.get_all_features()
    
    @staticmethod
    def get_default_features() -> List[str]:
        """获取默认启用的特征名称"""
        return FeatureEngineer.DEFAULT_FEATURES
    
    @staticmethod
    def set_default_features(features: List[str]):
        """设置默认特征列表"""
        FeatureEngineer.DEFAULT_FEATURES = features