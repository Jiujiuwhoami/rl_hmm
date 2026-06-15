#!/usr/bin/env python
"""Pipeline CLI — factor production and validation.

Usage:
    python pipeline/cli.py list-sets
    python pipeline/cli.py train --set st_features [--model-type xgboost] [OPTIONS]
    python pipeline/cli.py test --model st_features_xgboost --csv data.csv [OPTIONS]
    python pipeline/cli.py train --set core --model-type lightgbm --force
    python pipeline/cli.py train --set core --model-type rf --n_estimators 100 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure RL-GHMM root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.feature_set import list_feature_sets, load_feature_set


def cmd_list_sets(args):
    """List all available feature sets."""
    sets = list_feature_sets()
    if not sets:
        print("No feature sets found.")
        return

    print(f"Available feature sets ({len(sets)}):")
    print("-" * 60)
    for name in sets:
        try:
            fs = load_feature_set(name)
            print(f"  {name:<20s}  ({len(fs.features):>2d} features)  {fs.description}")
        except Exception as e:
            print(f"  {name:<20s}  [ERROR: {e}]")


def add_train_args(parser: argparse.ArgumentParser):
    """Add training arguments to a subparser."""
    parser.add_argument("--set", "-s", required=True,
                        help="Feature set name (e.g. st_features)")
    parser.add_argument("--model-type", "-t", default="xgboost",
                        choices=["xgboost", "lightgbm", "rf", "mlp", "logistic", "ensemble"],
                        help="Model type to train (default: xgboost)")
    parser.add_argument("--data_path", "-d", default="",
                        help="Training data CSV path (overrides YAML)")
    parser.add_argument("--db_path", default="", help="SQLite DB path")
    parser.add_argument("--exchange", default="", help="Exchange name")
    parser.add_argument("--symbol", default="", help="Trading pair")
    parser.add_argument("--start_time", default="", help="Start time (YYYY-MM-DD)")
    parser.add_argument("--end_time", default="", help="End time (YYYY-MM-DD)")
    parser.add_argument("--signal_interval", default="",
                        help="Signal timeframe override (e.g. 30m)")
    parser.add_argument("--early_minutes", type=int, default=None,
                        help="Early prediction minutes override")
    parser.add_argument("--models_dir", default="",
                        help="Model output directory (default: ../backtest-engine/strategies/models/)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing model")
    # Shared hyperparameters (used by xgboost, lightgbm, rf)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--win_reward", type=int, default=85)
    parser.add_argument("--loss_penalty", type=int, default=-100)
    # XGBoost / LightGBM shared
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=3.0)
    # LightGBM specific
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=20)
    parser.add_argument("--feature_fraction", type=float, default=0.8)
    parser.add_argument("--bagging_fraction", type=float, default=0.8)
    parser.add_argument("--bagging_freq", type=int, default=5)
    # Random Forest specific
    parser.add_argument("--min_samples_split", type=int, default=20)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--max_features", type=str, default="sqrt")
    # MLP specific
    parser.add_argument("--mlp_hidden", type=str, default="64,32",
                        help="MLP hidden layer sizes (comma-separated)")
    parser.add_argument("--mlp_activation", type=str, default="relu",
                        choices=["relu", "tanh", "logistic"])
    parser.add_argument("--mlp_solver", type=str, default="adam",
                        choices=["adam", "sgd", "lbfgs"])
    parser.add_argument("--mlp_alpha", type=float, default=0.0001)
    parser.add_argument("--mlp_max_iter", type=int, default=200)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_lr_init", type=float, default=0.001)
    # Logistic Regression specific
    parser.add_argument("--lr_C", type=float, default=1.0)
    parser.add_argument("--lr_penalty", type=str, default="l2",
                        choices=["l1", "l2", "elasticnet"])
    parser.add_argument("--lr_solver", type=str, default="lbfgs",
                        choices=["lbfgs", "liblinear", "saga"])
    parser.add_argument("--lr_max_iter", type=int, default=1000)
    # Ensemble specific
    parser.add_argument("--ensemble_method", type=str, default="voting",
                        choices=["voting", "stacking"],
                        help="Ensemble method (default: voting)")
    parser.add_argument("--ensemble_weights", type=str, default="",
                        help="Comma-separated voting weights, e.g. 1,1,1,2,1")


def add_test_args(parser: argparse.ArgumentParser):
    """Add testing arguments to a subparser."""
    parser.add_argument("--model", "-m", required=True,
                        help="Model name (feature set name, e.g. st_features)")
    parser.add_argument("--csv", "-c", default="",
                        help="Validation data CSV path (optional when using SQLite)")
    parser.add_argument("--output", "-o", default="",
                        help="Backtest output directory")
    parser.add_argument("--models_dir", default="",
                        help="Model directory (default: ../backtest-engine/strategies/models/)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override confidence threshold")
    parser.add_argument("--hold_bars", type=int, default=None,
                        help="Override hold bars")
    parser.add_argument("--db_path", default="", help="SQLite DB path")
    parser.add_argument("--exchange", default="", help="Exchange name")
    parser.add_argument("--symbol", default="", help="Trading pair")
    parser.add_argument("--start_time", default="", help="Start time (YYYY-MM-DD)")
    parser.add_argument("--end_time", default="", help="End time (YYYY-MM-DD)")


def main():
    parser = argparse.ArgumentParser(
        description="Factor Production & Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-sets
    p_list = sub.add_parser("list-sets", help="List available feature sets")
    p_list.set_defaults(func=cmd_list_sets)

    # train
    p_train = sub.add_parser("train", help="Train a model from a feature set")
    add_train_args(p_train)
    p_train.set_defaults(func=_run_train)

    # test
    p_test = sub.add_parser("test", help="Run backtest on a trained model")
    add_test_args(p_test)
    p_test.set_defaults(func=_run_test)

    args = parser.parse_args()
    args.func(args)


def _run_train(args):
    from pipeline.train import run_train
    run_train(args)


def _run_test(args):
    from pipeline.test import run_test
    run_test(args)


if __name__ == "__main__":
    main()
