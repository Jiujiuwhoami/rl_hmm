"""Train step — load feature set → train model → save model to models/{name}_{type}/."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure RL-GHMM root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.feature_set import load_feature_set

# Default model output directory (under RL-GHMM/models/)
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


_MODEL_TYPES = {
    'xgboost': {
        'label': 'XGBoost',
        'module': 'scripts.train_xgboost',
        'func': 'train_xgboost',
    },
    'lightgbm': {
        'label': 'LightGBM',
        'module': 'scripts.train_lightgbm',
        'func': 'train_lightgbm',
    },
    'rf': {
        'label': 'Random Forest',
        'module': 'scripts.train_random_forest',
        'func': 'train_random_forest',
    },
    'mlp': {
        'label': 'MLP Neural Network',
        'module': 'scripts.train_mlp',
        'func': 'train_mlp',
    },
    'logistic': {
        'label': 'Logistic Regression',
        'module': 'scripts.train_logistic',
        'func': 'train_logistic',
    },
    'ensemble': {
        'label': 'Ensemble (Voting/Stacking)',
        'module': 'scripts.train_ensemble',
        'func': 'train_ensemble',
    },
}


def _resolve_models_dir(overridden: str | None) -> Path:
    if overridden:
        return Path(overridden).resolve()
    return _DEFAULT_MODELS_DIR.resolve()


def run_train(args):
    """Run training for a feature set with the specified model type."""
    model_type = args.model_type
    if model_type not in _MODEL_TYPES:
        print(f"[ERROR] Unknown model type: {model_type}. Available: {list(_MODEL_TYPES.keys())}")
        return

    fs = load_feature_set(args.set)

    # CLI overrides
    signal_interval = args.signal_interval or fs.signal_interval
    early_minutes = args.early_minutes if args.early_minutes is not None else fs.early_minutes
    raw_data_path = args.data_path or fs.data_path
    data_path = str(_PROJECT_ROOT / raw_data_path) if not os.path.isabs(raw_data_path) else raw_data_path

    # Model output: {name}_{type}/model.pkl
    model_suffix = f"{fs.name}_{model_type}"
    models_dir = _resolve_models_dir(args.models_dir)
    target_dir = models_dir / model_suffix
    target_path = target_dir / "model.pkl"

    # Check if model already exists
    if target_path.exists() and not args.force:
        print(f"[SKIP] Model already exists: {target_path}")
        print(f"  Use --force to overwrite")
        return

    info = _MODEL_TYPES[model_type]
    print("=" * 70)
    print(f"[TRAIN] {info['label']}  |  Feature set: {fs.name}")
    print(f"  Description: {fs.description}")
    print(f"  Features ({len(fs.features)}): {fs.features}")
    print(f"  signal_interval: {signal_interval}, early_minutes: {early_minutes}")
    print(f"  data: {data_path}")
    print(f"  Output: {target_path}")
    print("=" * 70)

    # Import the training function dynamically
    mod = __import__(info['module'], fromlist=[info['func']])
    train_func = getattr(mod, info['func'])

    # Build kwargs from CLI args
    kwargs = {
        'data_path': data_path,
        'features': fs.features,
        'signal_interval': signal_interval,
        'early_minutes': early_minutes,
        'output_path': str(target_path),
        'train_ratio': args.train_ratio,
        'win_reward': args.win_reward,
        'loss_penalty': args.loss_penalty,
        'db_path': args.db_path or None,
        'exchange': args.exchange or None,
        'symbol': args.symbol or None,
        'start_time': args.start_time or None,
        'end_time': args.end_time or None,
    }

    # Model-specific hyperparams
    if model_type in ('xgboost', 'lightgbm'):
        kwargs.update(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
        )
    if model_type == 'xgboost':
        kwargs.update(
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )
    if model_type == 'lightgbm':
        kwargs.update(
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )
    if model_type == 'rf':
        kwargs.update(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth else 8,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
        )
    if model_type == 'mlp':
        kwargs.update(
            hidden_layer_sizes=tuple(int(x) for x in args.mlp_hidden.split(',')),
            activation=args.mlp_activation,
            solver=args.mlp_solver,
            alpha=args.mlp_alpha,
            max_iter=args.mlp_max_iter,
            batch_size=args.mlp_batch_size,
            learning_rate_init=args.mlp_lr_init,
        )
    if model_type == 'logistic':
        kwargs.update(
            C=args.lr_C,
            penalty=args.lr_penalty,
            solver=args.lr_solver,
            max_iter=args.lr_max_iter,
        )
    if model_type == 'ensemble':
        kwargs.update(
            method=args.ensemble_method,
            weights=[float(w) for w in args.ensemble_weights.split(',')] if args.ensemble_weights else None,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            rf_min_samples_split=args.min_samples_split,
            rf_min_samples_leaf=args.min_samples_leaf,
            mlp_hidden=tuple(int(x) for x in args.mlp_hidden.split(',')),
            mlp_activation=args.mlp_activation,
            mlp_alpha=args.mlp_alpha,
            mlp_max_iter=args.mlp_max_iter,
            mlp_batch_size=args.mlp_batch_size,
            mlp_lr_init=args.mlp_lr_init,
            lr_C=args.lr_C,
        )

    # Train
    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    success = train_func(**kwargs)
    elapsed = time.time() - t0

    if success is None:
        print(f"\n[ERROR] Training failed for '{model_suffix}'")
        return

    print(f"[DONE] {model_suffix} — time={elapsed:.1f}s")