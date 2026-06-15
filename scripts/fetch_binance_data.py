"""
Fetch Binance perpetual contract kline data, save to CSV or SQLite.

Usage:
    # Save to CSV (original)
    python fetch_binance_data.py --symbol BTCUSDT --start 2024-01-01 --end 2025-06-01 -o btc_5m.csv

    # Save to SQLite
    python fetch_binance_data.py --symbol BTCUSDT --start 2024-01-01 --end 2025-06-01 --db markets.db

    # Multiple intervals into same DB
    python fetch_binance_data.py --symbol BTCUSDT --interval 5m  --start 2024-01-01 --end 2025-06-01 --db markets.db
    python fetch_binance_data.py --symbol BTCUSDT --interval 15m --start 2024-01-01 --end 2025-06-01 --db markets.db
    python fetch_binance_data.py --symbol ETHUSDT --interval 5m  --start 2024-01-01 --end 2025-06-01 --db markets.db
"""
import argparse
import csv
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_time: int = None,
    end_time: int = None,
    limit: int = 1000,
) -> list:
    """
    Fetch kline data from Binance Futures API.
    
    Args:
        symbol: Trading pair (default: BTCUSDT)
        interval: Kline interval (default: 5m)
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Max number of records (max 1500)
    
    Returns:
        List of kline data
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1500),
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


def parse_klines(klines: list) -> list:
    """
    Parse Binance kline response to structured data.
    
    Binance kline format:
    [
        open_time, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_buy_volume,
        taker_buy_quote_volume, ignore
    ]
    """
    parsed = []
    for k in klines:
        parsed.append({
            "time": int(k[0]) // 1000,  # Convert ms to s
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return parsed


def fetch_range(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list:
    """
    Fetch all klines for a date range (handles pagination).
    """
    all_klines = []
    
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    print(f"Fetching {symbol} {interval} data from {start_dt} to {end_dt}")
    
    while start_ms < end_ms:
        print(f"  Fetching from {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)}...")
        
        klines = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_ms,
            end_time=end_ms,
            limit=1500,
        )
        
        if not klines:
            break
        
        parsed = parse_klines(klines)
        all_klines.extend(parsed)
        
        # Update start_time to last close_time + 1
        last_close_ms = klines[-1][6]  # close_time
        start_ms = last_close_ms + 1
        
        # Rate limit: 1200 requests per minute for IP
        time.sleep(0.05)
    
    return all_klines


def save_to_sqlite(data: list, db_path: str, exchange: str, symbol: str, interval: str):
    """Save kline data to SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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

    rows = [
        (exchange, symbol, interval, d["time"], d["open"], d["high"], d["low"], d["close"], d["volume"])
        for d in data
    ]

    inserted = 0
    skipped = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO kline (exchange, symbol, interval, time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
            if conn.total_changes > inserted + skipped:
                inserted += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1

    conn.commit()
    conn.close()

    if skipped:
        print(f"Saved {inserted} new records to {db_path} ({skipped} skipped — duplicates)")
    else:
        print(f"Saved {inserted} records to {db_path}")

    print(f"  [{exchange}] {symbol} ({interval}) — {inserted} new, {skipped} duplicate")


def save_to_csv(data: list, output_path: Path):
    """Save kline data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance futures kline data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="5m", help="Kline interval (default: 5m)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", default="", help="Output CSV path")
    parser.add_argument("--db", default="", help="SQLite database path (mutually exclusive with --output)")
    parser.add_argument("--exchange", default="binance", help="Exchange name for DB (default: binance)")

    args = parser.parse_args()

    if not args.output and not args.db:
        print("[ERROR] 必须提供 --output (CSV) 或 --db (SQLite)")
        return
    if args.output and args.db:
        print("[ERROR] --output 和 --db 不能同时使用，请选择一个")
        return

    # Parse dates
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Fetch data
    data = fetch_range(args.symbol, args.interval, start_dt, end_dt)

    if not data:
        print("[WARN] No data fetched")
        return

    # Save
    if args.db:
        save_to_sqlite(data, args.db, args.exchange, args.symbol, args.interval)
    else:
        save_to_csv(data, Path(args.output))

    # Print summary
    print(f"\nSummary:")
    print(f"  Total records: {len(data)}")
    print(f"  Start: {datetime.fromtimestamp(data[0]['time'], tz=timezone.utc)}")
    print(f"  End: {datetime.fromtimestamp(data[-1]['time'], tz=timezone.utc)}")
    print(f"  Price range: {min(d['low'] for d in data):.2f} - {max(d['high'] for d in data):.2f}")


if __name__ == "__main__":
    main()
