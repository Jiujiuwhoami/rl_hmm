# 尝试导入 stable_baselines3，如果失败则设为 None
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    DummyVecEnv = None
    EvalCallback = None
    SB3_AVAILABLE = False


class PPOAgent:
    """PPO强化学习Agent"""

    def __init__(self, vec_env, config):
        if not SB3_AVAILABLE:
            raise ImportError("stable_baselines3 is required for PPOAgent. Install with: pip install rl-hmm[gym]")
        
        self.config = config
        self.vec_env = vec_env
        self.model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=config.tensorboard_dir)

    def train(self):
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path=self.config.best_model_dir,
            log_path=self.config.log_dir,
            eval_freq=self.config.eval_freq
        )
        self.model.learn(total_timesteps=self.config.total_timesteps, callback=eval_callback)

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)