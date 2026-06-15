"""Batch train models for multiple symbols using SQLite data.
Saves to models/{symbol}/ to avoid overwriting existing models.

Usage:
    cd d:\myStateScoreSys
    python myStateScoreSysProd/RL-GHMM/scripts/batch_train.py
"""

import sys
import time
from pathlib import Path
from argparse import Namespace

# Ensure RL-GHMM root is on path
_RLGHMM = Path(__file__).resolve().parent.parent
if str(_RLGHMM) not in sys.path:
    sys.path.insert(0, str(_RLGHMM))

from pipeline.train import run_train

# === CONFIG ===
DB_PATH = _RLGHMM / "data" / "markets.db"
MODELS_BASE = _RLGHMM / "models"
EXCHANGE = "binance"
TRAIN_END = "2026-01-01"

# 6 symbols to train
SYMBOLS = ["BTCUSDT", "XRPUSDT", "HYPERUSDT", "DOGEUSDT", "BNBUSDT", "SOLUSDT"]

# 40 combos: 6 full feature sets × 6 model types + 4 single-model feature sets
FULL_SETS = ["core", "default_14feat", "momentum_vol", "no_bollinger", "no_macd", "rsi_macd"]
FULL_TYPES = ["xgboost", "lightgbm", "rf", "mlp", "logistic", "ensemble"]
SINGLE_SETS = ["momentum_only", "st_features", "trend_only", "volatility_only"]

# Model-specific defaults matching ETH training
MODEL_PARAMS = {
    "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "early_stopping_rounds": 20,
                 "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 3.0},
    "lightgbm": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "early_stopping_rounds": 20,
                  "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 3.0},
    "rf": {"n_estimators": 200, "max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": "sqrt"},
    "mlp": {"mlp_hidden": "64,32", "mlp_activation": "relu", "mlp_solver": "adam", "mlp_alpha": 0.0001,
             "mlp_max_iter": 200, "mlp_batch_size": 256, "mlp_lr_init": 0.001},
    "logistic": {"lr_C": 1.0, "lr_penalty": "l2", "lr_solver": "lbfgs", "lr_max_iter": 1000},
    "ensemble": {"ensemble_method": "voting", "ensemble_weights": "",
                  "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 3.0,
                  "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5,
                  "min_samples_split": 20, "min_samples_leaf": 10,
                  "mlp_hidden": "64,32", "mlp_activation": "relu", "mlp_alpha": 0.0001,
                  "mlp_max_iter": 200, "mlp_batch_size": 256, "mlp_lr_init": 0.001, "lr_C": 1.0},
}


def make_args(feat_set, model_type, symbol):
    """Build a mock argparse.Namespace for run_train()."""
    base = MODEL_PARAMS.get(model_type, {})
    args_dict = dict(base)
    args_dict.update({
        "set": feat_set,
        "model_type": model_type,
        "data_path": "",
        "db_path": str(DB_PATH),
        "exchange": EXCHANGE,
        "symbol": symbol,
        "start_time": None,
        "end_time": TRAIN_END,
        "signal_interval": "",
        "early_minutes": None,
        "models_dir": str(MODELS_BASE / symbol.split("USDT")[0].lower()),
        "force": False,
        "train_ratio": 0.8,
        "win_reward": 85,
        "loss_penalty": -100,
    })
    return Namespace(**args_dict)


def main():
    total_expected = len(SYMBOLS) * (len(FULL_SETS) * len(FULL_TYPES) + len(SINGLE_SETS))
    completed = 0
    skipped = 0
    errors = []
    overall_start = time.time()

    print(f"Batch training: {len(SYMBOLS)} symbols × {total_expected // len(SYMBOLS)} combos = {total_expected} models")
    print(f"Data source: SQLite ({DB_PATH})")
    print(f"Training data: up to {TRAIN_END}")
    print(f"Output: {MODELS_BASE}/{{symbol}}/")
    print("=" * 80)

    for symbol in SYMBOLS:
        symbol_lower = symbol.split("USDT")[0].lower()
        print(f"\n{'=' * 80}")
        print(f"SYMBOL: {symbol} ({symbol_lower})")
        print(f"{'=' * 80}")

        combos = []
        for fs in FULL_SETS:
            for mt in FULL_TYPES:
                combos.append((fs, mt))
        for fs in SINGLE_SETS:
            combos.append((fs, "xgboost"))

        for feat_set, model_type in combos:
            t0 = time.time()
            try:
                args = make_args(feat_set, model_type, symbol)
                # run_train will skip if model exists (no --force)
                run_train(args)
                elapsed = time.time() - t0
                completed += 1
                # Check if actually trained or skipped
                model_dir = MODELS_BASE / symbol_lower / f"{feat_set}_{model_type}"
                model_file = model_dir / "model.pkl"
                status = "TRAINED" if model_file.exists() else "SKIPPED"
                if status == "SKIPPED":
                    skipped += 1
                print(f"  [{completed}/{total_expected}] {symbol}/{feat_set}_{model_type} — {status} ({elapsed:.0f}s)\n")
            except Exception as e:
                elapsed = time.time() - t0
                errors.append(f"{symbol}/{feat_set}_{model_type}: {e}")
                print(f"  [{completed}/{total_expected}] {symbol}/{feat_set}_{model_type} — ERROR: {e} ({elapsed:.0f}s)\n")
                completed += 1

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 80}")
    print(f"DONE! Total: {total_expected}, Trained: {completed - skipped}, Skipped: {skipped}, Errors: {len(errors)}")
    print(f"Time: {overall_elapsed:.0f}s ({overall_elapsed/60:.1f}min)")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
