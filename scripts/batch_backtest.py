"""Batch backtest all models from SQLite and collect results."""
import ast, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "data/bt_results"
DB_PATH = ROOT / "data/markets.db"
EXCHANGE = "binance"
SYMBOL = "ETHUSDT"
START = "2026-01-01"
END = "2026-06-15"

results_dir = Path(RESULTS_DIR)
results_dir.mkdir(parents=True, exist_ok=True)

models = sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "model.pkl").exists()])
print(f"Models to backtest: {len(models)}")

all_results = {}

for i, model in enumerate(models, 1):
    out_dir = results_dir / model
    cmd = [
        sys.executable, str(ROOT / "pipeline/cli.py"), "test",
        "--model", model,
        "--db_path", str(DB_PATH),
        "--exchange", EXCHANGE,
        "--symbol", SYMBOL,
        "--start_time", START,
        "--end_time", END,
        "--output", str(out_dir),
    ]
    print(f"\n[{i}/{len(models)}] {model} ...", end=" ", flush=True)
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Parse summary
    summary_file = out_dir / "summary.txt"
    if summary_file.exists():
        text = summary_file.read_text()
        # Extract the dict from the line: "  xgboost_ml_strategy: {'total_trades': ...}"
        m = re.search(r"xgboost_ml_strategy: (\{.*\})", text)
        if m:
            raw = m.group(1)
            # Remove Timestamp(...) wrappers, keep just the string inside
            raw = re.sub(r"Timestamp\(['\"]([^'\"]+)['\"]\)", r"'\1'", raw)
            data = ast.literal_eval(raw)
            all_results[model] = data
            elapsed = time.time() - t0
            wr = data.get("win_rate", 0)
            pnl = data.get("total_pnl", 0)
            trades = data.get("total_trades", 0)
            print(f" trades={trades}, wr={wr:.1%}, pnl={pnl:.0f}, {elapsed:.0f}s")
        else:
            print(f" WARN: no summary dict in: {text[:100]}")
            all_results[model] = {"error": "parse_failed", "raw": text[:200]}
    else:
        print(f" ERROR")
        err = result.stderr[-300:] if result.stderr else "unknown"
        all_results[model] = {"error": err}

# Generate ranking
print("\n" + "=" * 80)
print(f"{'Rank':>4}  {'Model':<32}  {'Trades':>7}  {'Wins':>5}  {'WR':>6}  {'PnL':>10}  {'MDD':>6}")
print("=" * 80)

ranked = [(k, v) for k, v in all_results.items() if "total_pnl" in v]
ranked.sort(key=lambda x: x[1].get("total_pnl", 0), reverse=True)

for rank, (model, data) in enumerate(ranked, 1):
    trades = data.get("total_trades", 0)
    wins = data.get("win_trades", 0)
    wr = data.get("win_rate", 0)
    pnl = data.get("total_pnl", 0)
    mdd = data.get("max_drawdown", 0)
    print(f"{rank:>4}  {model:<32}  {trades:>7}  {wins:>5}  {wr:>5.1%}  {pnl:>+8.2f}  {mdd:>5.2%}")

# Save
out = results_dir / "_backtest_ranking.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nFull results saved to {out}")
print(f"Completed: {len(ranked)}/{len(models)}")
