# RL-HMM Trading System

Reinforcement Learning with Hidden Markov Models for Financial Trading.

## Features

- Integration of Hidden Markov Models (HMM) with Reinforcement Learning
- Multi-timeframe analysis support
- Feature engineering pipeline for financial data
- Backtesting framework for trading strategies

## Installation

```bash
pip install rl-hmm
```

## Usage

```python
from rl_hmm.system import TradingSystem

# Initialize trading system
system = TradingSystem()

# Run backtest
results = system.backtest()
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0

## License

MIT License
