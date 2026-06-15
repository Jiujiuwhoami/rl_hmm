"""Test (backtest) step — run backtest-engine on a trained model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import subprocess
import sys

# Ensure RL-GHMM root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl_hmm.data.loader import load_from_sqlite

_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"
_BACKTEST_CLI = _PROJECT_ROOT / "../backtest-engine/cli/main.py"
_DEFAULT_STRATEGY = _PROJECT_ROOT / "../backtest-engine/strategies/examples/xgboost_ml_strategy.py"


def _resolve_models_dir(overridden: str | None) -> Path:
    if overridden:
        return Path(overridden).resolve()
    return _DEFAULT_MODELS_DIR.resolve()


def run_test(args):
    """Run backtest for a trained model."""
    models_dir = _resolve_models_dir(args.models_dir)

    # Resolve model path
    model_path = _find_model(models_dir, args.model)
    if model_path is None:
        print(f"[ERROR] Model '{args.model}' not found in {models_dir}")
        print(f"  Tried: {models_dir / args.model / 'model.pkl'}")
        print(f"  Tried: {models_dir / f'{args.model}.pkl'}")
        sys.exit(1)

    print(f"[TEST] Model: {model_path}")

    # Load model metadata
    model_data = joblib.load(str(model_path))
    signal_interval = model_data.get("signal_interval", "15m")
    calib_thresh = model_data.get("calibrated_threshold", None)
    n_features = len(model_data.get("features", []))

    print(f"  signal_interval: {signal_interval}")
    print(f"  features: {n_features}")
    print(f"  calibrated_threshold: {calib_thresh}")

    # Resolve data source: SQLite > CSV
    csv_path = None
    if args.db_path and args.exchange and args.symbol:
        print(f"[DATA] Loading from SQLite: {args.exchange} {args.symbol}...")
        df = load_from_sqlite(
            args.db_path, args.exchange, args.symbol,
            interval='5m', start_time=args.start_time or None,
            end_time=args.end_time or None,
        )
        # Resample the DatetimeIndex back to Unix timestamps for CSV
        out_df = df.copy()
        out_df['open_time'] = (out_df.index.astype('int64') // 10**9).astype(int)
        out_df = out_df[['open_time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        out_df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        # Write to temp CSV for backtest CLI
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=str(_PROJECT_ROOT / "data"))
        out_df.to_csv(tmp.name, index=False)
        csv_path = Path(tmp.name)
        tmp.close()
        print(f"  OK {len(df)} rows -> {csv_path.name}")
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            sys.exit(1)
    else:
        print("[ERROR] Provide either --db_path/--exchange/--symbol or --csv")
        sys.exit(1)

    # Build params for the strategy
    params = f"model_name={args.model},model_path="
    if args.threshold is not None:
        params += f",confidence_threshold={args.threshold}"
    if args.hold_bars is not None:
        params += f",hold_bars={args.hold_bars}"

    # Build backtest CLI command
    bt_cli = _BACKTEST_CLI.resolve()
    strategy_path = _DEFAULT_STRATEGY.resolve()

    cmd = [
        sys.executable, str(bt_cli),
        "--strategy", str(strategy_path),
        "--csv", str(csv_path.resolve()),
        "--params", params,
    ]
    if args.output:
        cmd += ["--output", str(Path(args.output).resolve())]

    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    print()

    # Run backtest as subprocess
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)

    # Clean up temp CSV
    if csv_path and args.db_path:
        csv_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Backtest failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n[DONE] Backtest completed for model '{args.model}'")


def _find_model(models_dir: Path, name: str) -> Path | None:
    """Try to locate a model by name in the models directory."""
    # Priority: models/{name}/model.pkl > models/{name}.pkl
    candidate1 = models_dir / name / "model.pkl"
    if candidate1.exists():
        return candidate1
    candidate2 = models_dir / f"{name}.pkl"
    if candidate2.exists():
        return candidate2
    return None
