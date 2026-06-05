import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from ..config import Config
from ..hmm import MultiTimeframeHMM
from ..features import FeatureEngineer

# 尝试导入 gym，如果失败则设为 None
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False


# 如果 gym 可用，创建基础类
if GYM_AVAILABLE:
    _GymEnv = gym.Env
else:
    # 如果 gym 不可用，创建一个简单的替代基类
    class _GymEnv:
        """简化的环境基类（当 gym 不可用时）"""
        metadata = {'render.modes': []}
        action_space = None
        observation_space = None
        
        def __init__(self):
            pass
        
        def seed(self, seed=None):
            pass
        
        def step(self, action):
            raise NotImplementedError("需要 gym 才能使用 step 方法")
        
        def reset(self):
            raise NotImplementedError("需要 gym 才能使用 reset 方法")
        
        def render(self, mode='human'):
            pass


# ============ 策略接口（可插拔） ============

class RewardStrategy:
    """奖励策略接口"""
    
    def calculate(self, env: 'TradingEnv', action: int, actual_return: float) -> float:
        """计算奖励"""
        raise NotImplementedError


class ActionFilter:
    """动作过滤策略接口"""
    
    def filter(self, env: 'TradingEnv', action: int) -> int:
        """过滤/修改动作"""
        return action


class ObservationBuilder:
    """观察空间构建策略接口"""
    
    def build(self, env: 'TradingEnv') -> np.ndarray:
        """构建观察向量"""
        raise NotImplementedError


# ============ 默认实现 ============

class DefaultRewardStrategy(RewardStrategy):
    """默认奖励策略"""
    
    def calculate(self, env: 'TradingEnv', action: int, actual_return: float) -> float:
        config = env.config
        
        if action == 0:
            reward = -0.5
        else:
            pred_dir = 1 if action == 1 else -1
            actual_dir = 1 if actual_return > 0 else -1
            correct = pred_dir == actual_dir
            reward = config.win if correct else config.loss
            
            # 回撤惩罚
            if len(env.equity_curve) > 30:
                recent_max = max(env.equity_curve[-30:])
                dd = (recent_max - env.equity) / (recent_max + 1e-8)
                reward -= dd * 35
        
        return reward


class HMMActionFilter(ActionFilter):
    """基于HMM状态的动作过滤"""
    
    def filter(self, env: 'TradingEnv', action: int) -> int:
        if hasattr(env, 'hmm') and env.hmm is not None:
            regime = env.hmm.get_regime_filter(
                env.current_step, env.dfs, env.train_windows
            )
            if regime >= 1:
                return 0  # 强制跳过
        return action


class DefaultObservationBuilder(ObservationBuilder):
    """默认观察空间构建器"""
    
    def build(self, env: 'TradingEnv') -> np.ndarray:
        # 获取基础时间框架的数据
        base_interval = env.config.intervals[0]
        df_base = env.dfs.get(base_interval)
        
        if df_base is None or len(df_base) == 0:
            return np.zeros(len(env.feature_names), dtype=np.float32)
        
        window = df_base.iloc[max(0, env.current_step-60):env.current_step]
        
        # 动态获取特征值
        base_features = []
        for feature_name in env.feature_names:
            if feature_name in window.columns:
                base_features.append(window[feature_name].iloc[-1] if len(window) > 1 else 0)
            else:
                base_features.append(0.0)
        base_features = np.array(base_features)
        
        # 获取所有时间框架的HMM概率
        if hasattr(env, 'hmm') and env.hmm is not None:
            probs = env.hmm.get_all_probs(env.dfs, env.current_step, env.train_windows)
            return np.concatenate([base_features, probs]).astype(np.float32)
        
        return base_features.astype(np.float32)


class TradingEnv(_GymEnv):
    """RL交易环境 - 支持可插拔策略和多时间框架"""

    def __init__(
        self, 
        dfs: Dict[str, pd.DataFrame],
        config: Config,
        reward_strategy: Optional[RewardStrategy] = None,
        action_filter: Optional[ActionFilter] = None,
        observation_builder: Optional[ObservationBuilder] = None,
        use_hmm: bool = True
    ):
        if GYM_AVAILABLE:
            super().__init__()
        
        # 多时间框架数据：{时间框架: DataFrame}
        self.dfs = {k: v.reset_index(drop=True) for k, v in dfs.items()}
        self.config = config
        
        # 确保时间框架列表至少有一个元素
        if not config.intervals:
            self.intervals = ["15m"]
        else:
            self.intervals = config.intervals
        
        # 基础时间框架（最小粒度）
        self.base_interval = self.intervals[0]
        
        # 计算每个时间框架的训练窗口
        self.train_windows = self._calculate_train_windows()
        
        # 当前步骤（基于基础时间框架）
        df_base = self.dfs.get(self.base_interval)
        self.current_step = config.train_window if df_base is not None and len(df_base) > config.train_window else 0
        
        # 获取当前使用的特征列表
        self.feature_names = config.features if config.features is not None else FeatureEngineer.get_default_features()
        
        # 动作空间：0=跳过, 1=做多, 2=做空
        self.action_space = spaces.Discrete(3)
        
        # 计算观察空间维度：特征数 + HMM状态数 * 时间框架数
        obs_dim = len(self.feature_names)
        if use_hmm:
            obs_dim += len(self.intervals) * config.n_components
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # HMM模块（可选）
        self.hmm = MultiTimeframeHMM(config.n_components, self.intervals) if use_hmm else None
        
        # 策略组件（支持依赖注入）
        self.reward_strategy = reward_strategy or DefaultRewardStrategy()
        self.action_filter = action_filter or HMMActionFilter()
        self.observation_builder = observation_builder or DefaultObservationBuilder()
        
        # 状态变量
        self.equity = 0.0
        self.equity_curve = []
        self.trades = []
    
    def _calculate_train_windows(self) -> List[int]:
        """计算每个时间框架的训练窗口大小"""
        if self.config.train_windows is not None:
            return self.config.train_windows
        
        # 自动计算：按时间框架比例缩放
        base_window = self.config.train_window
        windows = []
        for i, interval in enumerate(self.intervals):
            if i == 0:
                windows.append(base_window)
            else:
                ratio = self._get_interval_ratio(interval)
                windows.append(max(100, base_window // ratio))
        return windows
    
    def _get_interval_ratio(self, interval: str) -> int:
        """获取时间框架相对于基础时间框架的倍数"""
        return self.config._parse_interval(interval) // self.config._parse_interval(self.base_interval)

    def reset(self):
        df_base = self.dfs.get(self.base_interval)
        self.current_step = self.config.train_window if df_base is not None and len(df_base) > self.config.train_window else 0
        self.equity = 0.0
        self.equity_curve = [0.0]
        self.trades = []
        if self.hmm:
            self.hmm.update(self.dfs, self.current_step, self.train_windows)
        return self._get_observation()

    def _get_observation(self):
        return self.observation_builder.build(self)

    def step(self, action):
        df_base = self.dfs.get(self.base_interval)
        done = df_base is None or self.current_step >= len(df_base) - 2
        
        if done:
            return self._get_observation(), 0, done, {}

        # 更新HMM（如果启用）
        if self.hmm:
            self.hmm.update(self.dfs, self.current_step, self.train_windows)

        # 应用动作过滤器
        action = self.action_filter.filter(self, action)

        # 获取实际收益（基于基础时间框架）
        if self.current_step + 1 < len(df_base):
            actual_return = df_base['return'].iloc[self.current_step + 1]
        else:
            actual_return = 0

        # 使用奖励策略计算奖励
        reward = self.reward_strategy.calculate(self, action, actual_return)

        # 更新状态
        self.equity += reward
        self.equity_curve.append(self.equity)
        self.trades.append(action)
        self.current_step += 1

        return self._get_observation(), reward, done, {"equity": self.equity}