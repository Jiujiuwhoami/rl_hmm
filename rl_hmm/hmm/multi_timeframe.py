import numpy as np
import pandas as pd
from typing import Dict, List
from .hmm_handler import HMMHandler


class MultiTimeframeHMM:
    """多时间框架HMM - 支持任意数量的时间框架"""

    def __init__(self, n_components: int = 3, intervals: List[str] = None):
        """
        Args:
            n_components: HMM组件数量
            intervals: 时间框架列表，按粒度从小到大排列
        """
        self.n_components = n_components
        self.intervals = intervals or ["15m", "1h"]
        # 存储每个时间框架的HMM处理器
        self.hmm_handlers: Dict[str, HMMHandler] = {
            interval: HMMHandler(n_components) for interval in self.intervals
        }
        
    def update(self, dfs: Dict[str, pd.DataFrame], current_step: int, train_windows: List[int]):
        """
        更新所有时间框架的HMM模型
        
        Args:
            dfs: {时间框架: DataFrame} 字典
            current_step: 当前步骤（基于最小时间框架）
            train_windows: 每个时间框架对应的训练窗口大小
        """
        for i, interval in enumerate(self.intervals):
            df = dfs.get(interval)
            if df is None or len(df) == 0:
                continue
            
            # 计算当前时间框架对应的索引
            ratio = self._get_ratio(i)
            idx = min(len(df) - 1, current_step // ratio)
            
            # 获取该时间框架的训练窗口
            window_size = train_windows[i] if train_windows else max(1, train_windows[0] // ratio)
            
            feat = self._prepare_features(df, idx, window_size)
            self.hmm_handlers[interval].fit(feat)

    def _prepare_features(self, df: pd.DataFrame, end_idx: int, window_size: int) -> np.ndarray:
        """准备特征数据"""
        start = max(0, end_idx - window_size)
        window = df.iloc[start:end_idx]
        if len(window) == 0:
            return np.zeros((1, 2))
        returns = window['return'].values.reshape(-1, 1)
        vol = pd.Series(returns.flatten()).rolling(20).std().fillna(0).values.reshape(-1, 1)
        return np.hstack([returns, vol])

    def get_regime_filter(self, current_step: int, dfs: Dict[str, pd.DataFrame], 
                          train_windows: List[int], target_interval: str = None) -> int:
        """
        获取指定时间框架的状态过滤
        
        Args:
            current_step: 当前步骤
            dfs: {时间框架: DataFrame} 字典
            train_windows: 训练窗口列表
            target_interval: 目标时间框架，默认使用最大的时间框架
            
        Returns:
            状态索引
        """
        # 默认使用最大的时间框架
        if target_interval is None:
            target_interval = self.intervals[-1] if self.intervals else "1h"
        
        if target_interval not in self.hmm_handlers:
            return 0
        
        idx = self.intervals.index(target_interval)
        ratio = self._get_ratio(idx)
        df = dfs.get(target_interval)
        
        if df is None or len(df) == 0:
            return 0
        
        df_idx = min(len(df) - 1, current_step // ratio)
        window_size = train_windows[idx] if train_windows else max(1, train_windows[0] // ratio)
        feat = self._prepare_features(df, df_idx, window_size)[-1:]
        
        return self.hmm_handlers[target_interval].predict(feat)
    
    def predict_proba(self, interval: str, features: np.ndarray) -> np.ndarray:
        """获取指定时间框架的状态概率"""
        handler = self.hmm_handlers.get(interval)
        if handler is None:
            return np.zeros(self.n_components)
        return handler.predict_proba(features)
    
    def get_all_probs(self, dfs: Dict[str, pd.DataFrame], current_step: int, 
                      train_windows: List[int]) -> np.ndarray:
        """获取所有时间框架的状态概率"""
        all_probs = []
        for i, interval in enumerate(self.intervals):
            df = dfs.get(interval)
            if df is None or len(df) == 0:
                all_probs.append(np.zeros(self.n_components))
                continue
            
            ratio = self._get_ratio(i)
            idx = min(len(df) - 1, current_step // ratio)
            window_size = train_windows[i] if train_windows else max(1, train_windows[0] // ratio)
            feat = self._prepare_features(df, idx, window_size)[-1:]
            probs = self.hmm_handlers[interval].predict_proba(feat)
            all_probs.append(probs)
        
        return np.concatenate(all_probs)
    
    def _get_ratio(self, idx: int) -> int:
        """获取第idx个时间框架相对于最小时间框架的倍数"""
        if idx == 0 or idx >= len(self.intervals):
            return 1
        return self._parse_interval(self.intervals[idx]) // self._parse_interval(self.intervals[0])
    
    def _parse_interval(self, interval: str) -> int:
        """解析时间间隔字符串为分钟数"""
        interval = interval.lower()
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 60 * 24
        return 1