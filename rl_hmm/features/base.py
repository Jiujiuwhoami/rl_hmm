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


@FeatureRegistry.register('bollinger_mid')
def feature_bollinger_mid(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """布林带中轨（移动平均）"""
    return df['Close'].rolling(window).mean()


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


# ============ ST 策略特征 ============

@FeatureRegistry.register('ema_60')
def feature_ema_60(df: pd.DataFrame) -> pd.Series:
    """60期指数移动平均"""
    return df['Close'].ewm(span=60, adjust=False).mean()


@FeatureRegistry.register('ema_720')
def feature_ema_720(df: pd.DataFrame) -> pd.Series:
    """720期指数移动平均"""
    return df['Close'].ewm(span=720, adjust=False).mean()


@FeatureRegistry.register('atr_stop_value')
def feature_atr_stop_value(df: pd.DataFrame, atr_period: int = 14, stoplossfactor: float = 5.0) -> pd.Series:
    """ATR止损价位（上轨或下轨，取决于价格方向）"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    # ATR
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0
    atr = pd.Series(tr).rolling(atr_period).mean().values

    basic_upper = close + atr * stoplossfactor
    basic_lower = close - atr * stoplossfactor
    close_ema = pd.Series(close).ewm(span=atr_period, adjust=False).mean().values
    direction = np.where(close >= close_ema, 1, -1)
    atr_stop = np.where(direction > 0, basic_lower, basic_upper)
    return pd.Series(atr_stop, index=df.index)


@FeatureRegistry.register('atr_stop_direction')
def feature_atr_stop_direction(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """ATR止损方向: 1=多头(价格在EMA上), -1=空头(价格在EMA下)"""
    close = df['Close'].values
    close_ema = pd.Series(close).ewm(span=atr_period, adjust=False).mean().values
    return pd.Series(np.where(close >= close_ema, 1, -1), index=df.index)


@FeatureRegistry.register('price_vs_atr_stop')
def feature_price_vs_atr_stop(df: pd.DataFrame, atr_period: int = 14, stoplossfactor: float = 5.0) -> pd.Series:
    """价格相对ATR止损位的归一化距离: (close - atr_stop) / atr"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0
    atr = pd.Series(tr).rolling(atr_period).mean()

    atr_vals = atr.values
    basic_upper = close + atr_vals * stoplossfactor
    basic_lower = close - atr_vals * stoplossfactor
    close_ema = pd.Series(close).ewm(span=atr_period, adjust=False).mean().values
    direction = np.where(close >= close_ema, 1, -1)
    atr_stop = np.where(direction > 0, basic_lower, basic_upper)

    return pd.Series((close - atr_stop) / np.maximum(atr_vals, 1e-10), index=df.index)


@FeatureRegistry.register('top_divergence')
def feature_top_divergence(df: pd.DataFrame, divergence_window: int = 60) -> pd.Series:
    """顶背离: 价格创局部新高但MACD柱未创新高"""
    close = df['Close'].values
    high = df['High'].values

    # MACD柱
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = pd.Series(macd - macd_signal, index=df.index)

    high_s = pd.Series(high, index=df.index)
    is_local_high = high_s.eq(
        high_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).max()
    )

    prev_high_price = high_s.where(is_local_high).ffill().shift(1)
    prev_high_hist = macd_hist.where(is_local_high).ffill().shift(1)

    return is_local_high & (high_s > prev_high_price) & (macd_hist < prev_high_hist)


@FeatureRegistry.register('bottom_divergence')
def feature_bottom_divergence(df: pd.DataFrame, divergence_window: int = 60) -> pd.Series:
    """底背离: 价格创局部新低但MACD柱未创新低"""
    close = df['Close'].values
    low = df['Low'].values

    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = pd.Series(macd - macd_signal, index=df.index)

    low_s = pd.Series(low, index=df.index)
    is_local_low = low_s.eq(
        low_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).min()
    )

    prev_low_price = low_s.where(is_local_low).ffill().shift(1)
    prev_low_hist = macd_hist.where(is_local_low).ffill().shift(1)

    return is_local_low & (low_s < prev_low_price) & (macd_hist > prev_low_hist)


@FeatureRegistry.register('trend_flag')
def feature_trend_flag(df: pd.DataFrame, divergence_window: int = 60) -> pd.Series:
    """趋势标记: 0=无, 1=底背离看涨, 2=顶背离看跌, 3=顶背离但价在上, 4=底背离但价在下"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = pd.Series(macd - macd_signal, index=df.index)

    high_s = pd.Series(high, index=df.index)
    low_s = pd.Series(low, index=df.index)
    close_s = pd.Series(close, index=df.index)

    is_local_high = high_s.eq(
        high_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).max()
    )
    is_local_low = low_s.eq(
        low_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).min()
    )

    prev_high_price = high_s.where(is_local_high).ffill().shift(1)
    prev_high_hist = macd_hist.where(is_local_high).ffill().shift(1)
    prev_low_price = low_s.where(is_local_low).ffill().shift(1)
    prev_low_hist = macd_hist.where(is_local_low).ffill().shift(1)

    top_div = is_local_high & (high_s > prev_high_price) & (macd_hist < prev_high_hist)
    bot_div = is_local_low & (low_s < prev_low_price) & (macd_hist > prev_low_hist)

    price_by_macd = pd.Series(np.nan, index=df.index, dtype='float64')
    price_by_macd.loc[top_div] = high_s[top_div]
    price_by_macd.loc[bot_div] = low_s[bot_div]
    price_by_macd = price_by_macd.ffill().fillna(close_s.rolling(divergence_window, min_periods=1).mean())

    flag = pd.Series(0, index=df.index, dtype='int64')
    flag.loc[bot_div & (close_s >= price_by_macd)] = 1
    flag.loc[top_div & (close_s <= price_by_macd)] = 2
    flag.loc[top_div & (close_s > price_by_macd)] = 3
    flag.loc[bot_div & (close_s < price_by_macd)] = 4
    flag = flag.replace(0, np.nan).ffill().fillna(0).astype('int64')
    return flag


@FeatureRegistry.register('price_by_macd')
def feature_price_by_macd(df: pd.DataFrame, divergence_window: int = 60) -> pd.Series:
    """MACD背离参考价（顶背离=高点价，底背离=低点价，向前填充）"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = pd.Series(macd - macd_signal, index=df.index)

    high_s = pd.Series(high, index=df.index)
    low_s = pd.Series(low, index=df.index)
    close_s = pd.Series(close, index=df.index)

    is_local_high = high_s.eq(
        high_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).max()
    )
    is_local_low = low_s.eq(
        low_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).min()
    )

    prev_high_price = high_s.where(is_local_high).ffill().shift(1)
    prev_high_hist = macd_hist.where(is_local_high).ffill().shift(1)
    prev_low_price = low_s.where(is_local_low).ffill().shift(1)
    prev_low_hist = macd_hist.where(is_local_low).ffill().shift(1)

    top_div = is_local_high & (high_s > prev_high_price) & (macd_hist < prev_high_hist)
    bot_div = is_local_low & (low_s < prev_low_price) & (macd_hist > prev_low_hist)

    ref = pd.Series(np.nan, index=df.index, dtype='float64')
    ref.loc[top_div] = high_s[top_div]
    ref.loc[bot_div] = low_s[bot_div]
    ref = ref.ffill().fillna(close_s.rolling(divergence_window, min_periods=1).mean())
    return ref


@FeatureRegistry.register('price_vs_macd_ref')
def feature_price_vs_macd_ref(df: pd.DataFrame, divergence_window: int = 60) -> pd.Series:
    """价格相对MACD背离参考价的归一化距离: (close - ref) / close"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    close_s = pd.Series(close, index=df.index)

    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = pd.Series(macd - macd_signal, index=df.index)

    high_s = pd.Series(high, index=df.index)
    low_s = pd.Series(low, index=df.index)

    is_local_high = high_s.eq(
        high_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).max()
    )
    is_local_low = low_s.eq(
        low_s.rolling(divergence_window + 1, min_periods=divergence_window + 1).min()
    )

    prev_high_price = high_s.where(is_local_high).ffill().shift(1)
    prev_high_hist = macd_hist.where(is_local_high).ffill().shift(1)
    prev_low_price = low_s.where(is_local_low).ffill().shift(1)
    prev_low_hist = macd_hist.where(is_local_low).ffill().shift(1)

    top_div = is_local_high & (high_s > prev_high_price) & (macd_hist < prev_high_hist)
    bot_div = is_local_low & (low_s < prev_low_price) & (macd_hist > prev_low_hist)

    ref = pd.Series(np.nan, index=df.index, dtype='float64')
    ref.loc[top_div] = high_s[top_div]
    ref.loc[bot_div] = low_s[bot_div]
    ref = ref.ffill().fillna(close_s.rolling(divergence_window, min_periods=1).mean())

    return pd.Series((close_s - ref) / close_s.replace(0, np.nan), index=df.index)


@FeatureRegistry.register('vol_ma_60')
def feature_vol_ma_60(df: pd.DataFrame) -> pd.Series:
    """60期成交量移动平均"""
    return df['Volume'].rolling(60).mean()


@FeatureRegistry.register('vol_prev_ratio')
def feature_vol_prev_ratio(df: pd.DataFrame) -> pd.Series:
    """成交量/前一根成交量"""
    return df['Volume'] / df['Volume'].shift(1).replace(0, np.nan)


@FeatureRegistry.register('volume_filter_strict')
def feature_volume_filter_strict(df: pd.DataFrame, vol_ma_window: int = 60,
                                   vol_ma_multiplier: float = 2.0,
                                   vol_prev_multiplier: float = 2.0) -> pd.Series:
    """严格成交量过滤: 当前量>=均量*2 且 >=前一根量*2"""
    vol = df['Volume']
    vol_ma = vol.rolling(vol_ma_window).mean()
    return (vol >= vol_ma * vol_ma_multiplier) & (vol >= vol.shift(1) * vol_prev_multiplier)


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