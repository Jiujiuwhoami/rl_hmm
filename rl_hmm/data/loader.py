import pandas as pd
from ..features import FeatureEngineer


class DataLoader:
    """数据加载器 - 支持动态特征配置"""

    def __init__(self, config):
        self.config = config

    def load_data(self):
        """
        默认返回空的DataFrame
        
        用户需要自行提供数据，可以通过以下方式：
        1. 继承 DataLoader 并重写 load_data 方法
        2. 在外部加载数据后直接传入 TradingEnv
        
        Returns:
            (df_15m, df_1h): 两个空的DataFrame
        """
        df_15m = pd.DataFrame()
        df_1h = pd.DataFrame()
        return df_15m, df_1h