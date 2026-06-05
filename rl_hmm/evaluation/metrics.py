import pandas as pd
import matplotlib.pyplot as plt


class Evaluator:
    """评估器"""

    def __init__(self, config):
        self.config = config

    def evaluate(self, model, test_env):
        obs = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = test_env.step(action)

        equity = pd.Series(test_env.equity_curve)
        total_trades = sum(1 for a in test_env.trades if a != 0)
        wins = sum(1 for i in range(1, len(equity)) if equity.iloc[i] > equity.iloc[i-1] and test_env.trades[i-1] != 0)
        win_rate = wins / total_trades if total_trades > 0 else 0
        max_dd = ((equity.cummax() - equity) / equity.cummax().replace(0, 1)).max()

        print("\n=== 回测结果 ===")
        print(f"最终积分: {equity.iloc[-1]:.0f}")
        print(f"交易次数: {total_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"最大回撤: {max_dd:.2%}")

        return equity

    def plot_results(self, df_price, equity):
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df_price['Close'].values[self.config.train_window:], label='Price')
        plt.title(f"{self.config.ticker} 15m Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(equity, label='Equity Curve', color='green')
        plt.title('HMM + PPO Equity Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()