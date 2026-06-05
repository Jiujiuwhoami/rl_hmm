from stable_baselines3.common.vec_env import DummyVecEnv
from ..config import Config
from ..data import DataLoader
from ..environment import TradingEnv
from ..agents import PPOAgent
from ..evaluation import Evaluator


class TradingSystem:
    """RL-HMM交易系统主类"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.evaluator = Evaluator(self.config)

    def run(self):
        df_15m, df_1h = self.data_loader.load_data()

        print(f"创建 {self.config.n_env} 个并行环境...")
        def make_env():
            return TradingEnv(df_15m.copy(), df_1h.copy(), self.config)

        vec_env = DummyVecEnv([make_env for _ in range(self.config.n_env)])

        print("开始 RL 训练 (PPO)...")
        agent = PPOAgent(vec_env, self.config)
        agent.train()

        test_env = TradingEnv(df_15m, df_1h, self.config)
        equity = self.evaluator.evaluate(agent, test_env)
        self.evaluator.plot_results(test_env.df_15m, equity)