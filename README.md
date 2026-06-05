# RL-HMM 交易系统库

[![PyPI version](https://badge.fury.io/py/rl-hmm.svg)](https://badge.fury.io/py/rl-hmm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个模块化的强化学习结合隐马尔可夫模型的交易系统库，支持多时间框架分析。

## 安装

### 从 PyPI 安装（推荐）

```bash
pip install rl-hmm
```

### 安装完整版本（含强化学习依赖）

```bash
pip install rl-hmm[gym]
```

### 从源码安装

```bash
git clone https://github.com/yourusername/rl-hmm.git
cd rl-hmm
pip install -e .
```

## 功能特性

- **多时间框架支持**：支持 5分钟、15分钟、1小时等多种时间框架
- **特征工程**：内置多种技术指标（RSI、MACD、ATR、布林带等）
- **HMM状态建模**：隐马尔可夫模型捕捉市场状态
- **强化学习**：支持 PPO 算法的交易策略学习
- **模块化设计**：灵活配置特征、奖励机制、动作空间等

## 项目结构

```
rl_hmm/
├── __init__.py
├── config/              # 配置模块
│   ├── __init__.py
│   └── default_config.py
├── features/            # 特征工程模块
│   ├── __init__.py
│   └── base.py
├── hmm/                 # HMM模块
│   ├── __init__.py
│   ├── hmm_handler.py
│   └── multi_timeframe.py
├── environment/         # Gym环境模块
│   ├── __init__.py
│   └── trading_env.py
├── agents/              # RL Agent模块
│   ├── __init__.py
│   └── ppo_agent.py
├── data/                # 数据加载模块
│   ├── __init__.py
│   └── loader.py
├── evaluation/          # 评估模块
│   ├── __init__.py
│   └── metrics.py
└── system/              # 交易系统主类
    ├── __init__.py
    └── trading_system.py
```

## 快速开始

### 基础使用

```python
from rl_hmm import TradingSystem, Config

config = Config()
system = TradingSystem(config)
system.run()
```

### 自定义配置

```python
from rl_hmm import TradingSystem, Config

config = Config()
config.ticker = "AAPL"
config.total_timesteps = 100000
config.n_components = 4
config.win = 100
config.loss = -80

system = TradingSystem(config)
system.run()
```

### 模块化使用

```python
from rl_hmm import Config, DataLoader, FeatureEngineer, HMMHandler
import numpy as np

config = Config()
loader = DataLoader(config)
df_15m, df_1h = loader.load_data()

hmm = HMMHandler(config.n_components)
features = df_15m['return'].values.reshape(-1, 1)
hmm.fit(features)
prediction = hmm.predict(features[-1:])
```

## 核心模块说明

### Config
配置类，包含所有系统参数：
- ticker: 交易代码
- interval_15m: 15分钟K线间隔
- interval_1h: 1小时K线间隔
- train_window: 训练窗口大小
- total_timesteps: RL训练总步数
- n_components: HMM组件数量
- n_env: 并行环境数量
- win: 获胜奖励
- loss: 失败惩罚

### FeatureEngineer
特征工程类，提供技术指标提取功能：
- add_all_features(): 添加所有特征
- get_feature_names(): 获取特征名称列表

### HMMHandler
单个HMM处理器：
- fit(): 训练HMM模型
- predict(): 预测状态
- predict_proba(): 预测状态概率

### MultiTimeframeHMM
多时间框架HMM处理器：
- update(): 更新多时间框架HMM模型
- get_regime_filter(): 获取市场状态过滤

### TradingEnv
Gym交易环境，用于RL训练：
- reset(): 重置环境
- step(): 执行一步交易
- _get_observation(): 获取观察状态

### PPOAgent
PPO强化学习Agent：
- train(): 训练Agent
- predict(): 预测动作
- save(): 保存模型
- load(): 加载模型

### DataLoader
数据加载器：
- load_data(): 下载并加载数据

### Evaluator
评估器：
- evaluate(): 评估模型性能
- plot_results(): 绘制结果

### TradingSystem
交易系统主类，协调所有模块：
- run(): 运行完整的交易系统

## 特点

1. 模块化设计：各功能模块独立，便于扩展和维护
2. 面向对象：清晰的类结构和职责划分
3. 易于调用：提供简单易用的API
4. 多时间框架：支持多时间框架分析
5. 灵活配置：可自定义各种参数

## 依赖项

- numpy
- pandas
- yfinance
- hmmlearn
- gym
- stable-baselines3
- pandas-ta
- matplotlib

## 许可证

MIT License