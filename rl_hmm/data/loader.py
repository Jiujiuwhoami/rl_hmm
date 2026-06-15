import pandas as pd
import sqlite3
from typing import Optional
from ..features import FeatureEngineer


def load_from_sqlite(
    db_path: str,
    exchange: str,
    symbol: str,
    interval: str = '5m',
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """从 SQLite 数据库加载 K 线数据。

    Args:
        db_path: SQLite 数据库文件路径
        exchange: 交易所名称（如 binance, okx）
        symbol: 交易对（如 BTCUSDT, ETHUSDT）
        interval: K 线周期（如 5m, 15m, 1h）
        start_time: 开始时间 (可选，格式 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS')
        end_time: 结束时间 (可选，格式同上)

    Returns:
        DataFrame 包含 DatetimeIndex 和 Open/High/Low/Close/Volume 列
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT time, open, high, low, close, volume
        FROM kline
        WHERE exchange = ? AND symbol = ? AND interval = ?
    """
    params: list = [exchange, symbol, interval]

    if start_time:
        try:
            ts = int(start_time)
            query += " AND time >= ?"
            params.append(ts)
        except ValueError:
            query += " AND time >= ?"
            params.append(int(pd.Timestamp(start_time).timestamp()))
    if end_time:
        try:
            ts = int(end_time)
            query += " AND time <= ?"
            params.append(ts)
        except ValueError:
            query += " AND time <= ?"
            params.append(int(pd.Timestamp(end_time).timestamp()))

    query += " ORDER BY time ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        raise ValueError(
            f"SQLite 查询结果为空: db={db_path}, exchange={exchange}, "
            f"symbol={symbol}, interval={interval}"
        )

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def import_csv_to_sqlite(
    csv_path: str,
    db_path: str,
    exchange: str,
    symbol: str,
    interval: str = '5m',
) -> int:
    """将 CSV 文件导入 SQLite 数据库。

    CSV 需包含列: time(或timestamp), open, high, low, close, volume
    time 可以是 Unix 时间戳或日期字符串。

    Returns:
        导入的行数
    """
    import numpy as np

    df = pd.read_csv(csv_path)

    # 自动探测 time 列名
    time_cols = [c for c in df.columns if c.lower() in ('time', 'timestamp', 'date', 'datetime')]
    if time_cols:
        time_col = time_cols[0]
    else:
        raise ValueError("CSV 中找不到 time/timestamp/date/datetime 列")

    # 统一列名
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('time', 'timestamp', 'date', 'datetime'):
            rename_map[c] = 'time'
        elif cl in ('open', 'o'):
            rename_map[c] = 'open'
        elif cl in ('high', 'h'):
            rename_map[c] = 'high'
        elif cl in ('low', 'l'):
            rename_map[c] = 'low'
        elif cl in ('close', 'c'):
            rename_map[c] = 'close'
        elif cl in ('volume', 'vol', 'v'):
            rename_map[c] = 'volume'
    df.rename(columns=rename_map, inplace=True)

    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 转换 time 为 Unix 时间戳（秒）
    if df['time'].dtype in (np.int64, np.float64):
        # 已经是数字，判断是秒还是毫秒
        if df['time'].max() > 1e12:  # 毫秒 -> 秒
            df['time'] = (df['time'] // 1000).astype(int)
        else:
            df['time'] = df['time'].astype(int)
    else:
        df['time'] = pd.to_datetime(df['time']).astype(int) // 10**9

    df['exchange'] = exchange
    df['symbol'] = symbol
    df['interval'] = interval

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(exchange, symbol, interval, time)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_lookup ON kline(exchange, symbol, interval, time)")

    n = df[['exchange', 'symbol', 'interval', 'time', 'open', 'high', 'low', 'close', 'volume']].to_sql(
        'kline', conn, if_exists='append', index=False, method='multi'
    )
    conn.commit()
    conn.close()

    print(f"  [{exchange}] {symbol} {interval}: {n} rows imported to {db_path}")
    return n


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