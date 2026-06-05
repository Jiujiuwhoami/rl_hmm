#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL-GHMM 使用示例集

展示不同的使用方式和配置选项
"""

import numpy as np
import pandas as pd


# ============================================================
# 示例1: 基础使用 - 使用默认配置
# ============================================================
def example_1_basic_usage():
    """
    最简单的使用方式，使用所有默认配置
    """
    from rl_hmm import TradingSystem, Config

    config = Config()
    config.ticker = "BTC"
    config.total_timesteps = 50000
    
    system = TradingSystem(config)
    # system.run()  # 取消注释以运行


# ============================================================
# 示例2: 多时间框架配置
# ============================================================
def example_2_multi_timeframe():
    """
    配置多个时间框架进行交易
    支持任意数量: ["1m", "5m", "15m", "1h", "4h", "1d"]
    """
    from rl_hmm import TradingSystem, Config

    config = Config()
    config.ticker = "BTC"
    config.total_timesteps = 100000
    
    # 配置多时间框架
    config.intervals = ["5m", "15m", "1h"]  # 三个时间框架
    
    # 可选：手动设置每个时间框架的训练窗口
    # config.train_windows = [1000, 200, 50]
    
    system = TradingSystem(config)
    # system.run()


# ============================================================
# 示例3: 自定义特征列表
# ============================================================
def example_3_custom_features():
    """
    选择性地使用部分特征，减少观察空间维度
    """
    from rl_hmm import TradingSystem, Config
    from rl_hmm.features import FeatureEngineer

    config = Config()
    config.ticker = "BTC"
    config.total_timesteps = 50000
    
    # 只使用部分特征
    config.features = ['return', 'rsi', 'atr']
    
    print(f"使用的特征: {config.features}")
    print(f"所有可用特征: {FeatureEngineer.get_available_features()}")
    
    system = TradingSystem(config)
    # system.run()


# ============================================================
# 示例4: 动态注册新特征
# ============================================================
def example_4_register_feature():
    """
    在不修改源码的情况下添加新特征
    """
    from rl_hmm import Config, FeatureRegistry

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

    config = Config()
    config.features = ['return', 'rsi', 'price_to_sma_ratio', 'momentum']
    
    print(f"包含自定义特征: {config.features}")
    
    # system = TradingSystem(config)
    # system.run()


# ============================================================
# 示例5: 自定义奖励策略
# ============================================================
def example_5_custom_reward():
    """
    自定义奖励计算逻辑
    """
    from rl_hmm import Config, TradingEnv, RewardStrategy

    class MyRewardStrategy(RewardStrategy):
        """自定义奖励策略"""
        
        def calculate(self, env, action, actual_return):
            config = env.config
            
            if action == 0:  # 跳过
                return -0.1
            
            # 预测方向
            pred_dir = 1 if action == 1 else -1
            actual_dir = 1 if actual_return > 0 else -1
            
            if pred_dir == actual_dir:
                # 正确预测，获得奖励
                return config.win
            else:
                # 错误预测，惩罚
                return config.loss

    config = Config()
    config.ticker = "BTC"
    
    # 创建环境时注入自定义奖励策略
    # env = TradingEnv(dfs, config, reward_strategy=MyRewardStrategy())


# ============================================================
# 示例6: 自定义动作过滤
# ============================================================
def example_6_custom_filter():
    """
    自定义动作过滤逻辑，例如基于波动率过滤
    """
    from rl_hmm import Config, TradingEnv, ActionFilter

    class VolatilityFilter(ActionFilter):
        """基于波动率的动作过滤"""
        
        def filter(self, env, action):
            if action == 0:
                return action
            
            # 获取当前波动率
            if 'volatility' in env.feature_names:
                df_base = env.dfs.get(env.base_interval)
                volatility = df_base['volatility'].iloc[env.current_step]
                
                # 高波动时禁止交易
                if volatility > 0.02:
                    return 0
            
            return action

    config = Config()
    # env = TradingEnv(dfs, config, action_filter=VolatilityFilter())


# ============================================================
# 示例7: 分步使用（完全控制）
# ============================================================
def example_7_step_by_step():
    """
    分步使用各个组件，获得完全的控制权
    """
    from rl_hmm import Config, FeatureEngineer, TradingEnv
    from rl_hmm import PPOAgent, Evaluator

    # 1. 配置
    config = Config()
    config.ticker = "BTC"
    config.intervals = ["5m", "15m"]
    config.train_window = 500
    config.total_timesteps = 50000
    config.features = ['return', 'rsi', 'atr', 'volatility']

    # 2. 加载和处理数据
    # df_5m = pd.read_csv('btc_5m.csv')
    # df_15m = pd.read_csv('btc_15m.csv')
    
    # 模拟数据
    n = 1000
    df = pd.DataFrame({
        'Open': np.random.randn(n) + 100,
        'High': np.random.randn(n) + 102,
        'Low': np.random.randn(n) + 98,
        'Close': np.random.randn(n) + 100,
        'Volume': np.random.randint(1000, 10000, n)
    })
    df_5m = df_15m = df
    
    # 添加特征
    df_5m = FeatureEngineer.add_features(df_5m, config.features)
    df_15m = FeatureEngineer.add_features(df_15m, config.features)
    
    # 构建多时间框架数据字典
    dfs = {"5m": df_5m, "15m": df_15m}

    # 3. 创建环境
    env = TradingEnv(dfs, config)
    
    # 4. 创建Agent进行训练
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # vec_env = DummyVecEnv([lambda: TradingEnv(dfs.copy(), config)])
    # agent = PPOAgent(vec_env, config)
    # agent.train()

    # 5. 评估
    # evaluator = Evaluator(config)
    # equity = evaluator.evaluate(agent, env)

    print("分步使用流程完成")
    print(f"环境配置: {config.intervals}")
    print(f"特征列表: {config.features}")


# ============================================================
# 示例8: 禁用HMM
# ============================================================
def example_8_without_hmm():
    """
    不使用HMM，只用特征进行训练
    """
    from rl_hmm import Config, TradingEnv

    config = Config()
    config.ticker = "BTC"
    config.features = ['return', 'rsi', 'atr', 'volatility', 'macd_hist']
    
    # 禁用HMM，观察空间会变小
    # env = TradingEnv(dfs, config, use_hmm=False)
    
    # 观察空间维度 = 特征数量（无HMM概率）
    # obs_dim = len(config.features)


# ============================================================
# 示例9: 完整自定义
# ============================================================
def example_9_full_customization():
    """
    完全自定义所有策略
    """
    from rl_hmm import (
        Config, TradingEnv, 
        RewardStrategy, ActionFilter, ObservationBuilder
    )

    class CustomReward(RewardStrategy):
        def calculate(self, env, action, actual_return):
            # 自定义奖励逻辑
            return 1.0 if action != 0 else -0.1

    class CustomFilter(ActionFilter):
        def filter(self, env, action):
            # 自定义过滤逻辑
            return action

    class CustomObsBuilder(ObservationBuilder):
        def build(self, env):
            # 自定义观察空间
            df = env.dfs.get(env.base_interval)
            return np.array([
                df['return'].iloc[env.current_step],
                df['rsi'].iloc[env.current_step]
            ], dtype=np.float32)

    config = Config()
    # env = TradingEnv(dfs, config,
    #                  reward_strategy=CustomReward(),
    #                  action_filter=CustomFilter(),
    #                  observation_builder=CustomObsBuilder())


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("RL-GHMM 示例集")
    print("=" * 60)
    
    examples = [
        ("基础使用", example_1_basic_usage),
        ("多时间框架", example_2_multi_timeframe),
        ("自定义特征", example_3_custom_features),
        ("动态注册特征", example_4_register_feature),
        ("自定义奖励策略", example_5_custom_reward),
        ("自定义动作过滤", example_6_custom_filter),
        ("分步使用", example_7_step_by_step),
        ("禁用HMM", example_8_without_hmm),
        ("完全自定义", example_9_full_customization),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n示例 {i}: {name}")
        print("-" * 40)
        try:
            func()
        except Exception as e:
            print(f"  [跳过] 需要运行环境: {type(e).__name__}")

    print("\n" + "=" * 60)
    print("所有示例执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
