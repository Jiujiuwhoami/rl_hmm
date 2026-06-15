#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV-to-SQLite 导入工具

将 CSV 格式的 K 线数据导入 SQLite 数据库。
支持批量导入多个 CSV 文件，自动建表。

Usage:
    # 导入单个文件
    python scripts/import_to_sqlite.py --csv btc_5m.csv --db markets.db --exchange binance --symbol BTCUSDT

    # 指定周期（默认 5m）
    python scripts/import_to_sqlite.py --csv eth_15m.csv --db markets.db --exchange binance --symbol ETHUSDT --interval 15m

    # 批量导入（多次调用即可）
    python scripts/import_to_sqlite.py --csv btc_5m.csv --db markets.db --exchange binance --symbol BTCUSDT
    python scripts/import_to_sqlite.py --csv eth_5m.csv --db markets.db --exchange binance --symbol ETHUSDT

    # 导入后可以直接训练，不再需要 --data_path
    python scripts/train_xgboost.py --db_path markets.db --exchange binance --symbol BTCUSDT --features return,rsi,macd
"""

import argparse
import sys
from pathlib import Path

# Ensure RL-GHMM root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl_hmm.data.loader import import_csv_to_sqlite


def main():
    parser = argparse.ArgumentParser(
        description="将 CSV K 线数据导入 SQLite 数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/import_to_sqlite.py --csv btc_5m.csv --db markets.db --exchange binance --symbol BTCUSDT
  python scripts/import_to_sqlite.py --csv eth_5m.csv --db markets.db --exchange binance --symbol ETHUSDT --interval 5m
        """,
    )
    parser.add_argument("--csv", "-c", required=True, help="CSV 文件路径")
    parser.add_argument("--db", "-d", required=True, help="SQLite 数据库路径")
    parser.add_argument("--exchange", "-e", required=True, help="交易所名称 (如 binance, okx)")
    parser.add_argument("--symbol", "-s", required=True, help="交易对 (如 BTCUSDT, ETHUSDT)")
    parser.add_argument("--interval", "-i", default="5m", help="K 线周期 (默认 5m)")

    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[ERROR] CSV 文件不存在: {args.csv}")
        return

    print(f"开始导入...")
    print(f"  CSV:      {args.csv}")
    print(f"  DB:       {args.db}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Interval: {args.interval}")

    n = import_csv_to_sqlite(args.csv, args.db, args.exchange, args.symbol, args.interval)
    print(f"[DONE] 成功导入 {n} 条数据到 {args.db}")


if __name__ == "__main__":
    main()