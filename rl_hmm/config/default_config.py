class Config:
    """RL-HMM 交易系统配置类"""

    def __init__(self):
        self.ticker = "AAPL"
        # 多时间框架配置：按粒度从小到大排列
        # 支持任意数量的时间框架，例如: ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.intervals = ["15m", "1h"]
        # 每个时间框架对应的训练窗口大小（可选，不设置则自动计算）
        self.train_windows = None  # 格式: [2000, 500] 对应每个interval
        self.train_window = 2000  # 默认训练窗口（用于自动计算）
        self.total_timesteps = 200000
        self.n_components = 3
        self.n_env = 4
        self.win = 85
        self.loss = -100
        self.eval_freq = 20000
        self.log_dir = "./logs/"
        self.best_model_dir = "./best_model/"
        self.tensorboard_dir = "./hmm_rl_tensorboard/"
        # 特征配置：None表示使用默认特征列表
        self.features = None
        
    def get_interval_ratio(self, interval: str) -> int:
        """获取时间框架相对于最小时间框架的倍数"""
        if not self.intervals:
            return 1
        min_interval = self.intervals[0]
        return self._parse_interval(interval) // self._parse_interval(min_interval)
    
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