#!/usr/bin/env python
"""
XGBoost Feature Selection Experiments
======================================
Tests multiple feature combinations with identical XGBoost hyperparams
to identify the best-performing feature set.
"""
import sys
sys.path.insert(0, 'd:/myStateScoreSys/RL-GHMM')
sys.path.insert(0, 'd:/myStateScoreSys/RL-GHMM/rl_hmm')
import subprocess
import json
import time
import os

# Base command
BASE_CMD = [
    sys.executable, 'train_xgboost_profitable.py',
    '--signal_interval', '15m',
    '--early_minutes', '0',
    '--data_path', 'btc_5m_20240401_20260111.csv',
]

# Feature groups
MOMENTUM = ['return', 'rsi', 'macd', 'macd_signal', 'macd_hist']
VOLATILITY = ['volatility', 'atr', 'bollinger_upper', 'bollinger_lower', 'vol_change']
TREND = ['sma_20', 'ema_20', 'sma_60', 'vol_ma_ratio']
ALL_FEATURES = MOMENTUM + VOLATILITY + TREND  # 14 features

# Feature combinations to test
EXPERIMENTS = [
    ('01_all', ALL_FEATURES),
    ('02_momentum_only', MOMENTUM),
    ('03_volatility_only', VOLATILITY),
    ('04_trend_only', TREND),
    ('05_momentum_vol', MOMENTUM + VOLATILITY),
    ('06_momentum_trend', MOMENTUM + TREND),
    ('07_vol_trend', VOLATILITY + TREND),
    ('08_no_bollinger', [f for f in ALL_FEATURES if f not in ('bollinger_upper', 'bollinger_lower')]),
    ('09_no_macd', [f for f in ALL_FEATURES if f not in ('macd', 'macd_signal', 'macd_hist')]),
    ('10_core', ['return', 'rsi', 'macd', 'macd_hist', 'volatility', 'atr', 'sma_20', 'ema_20', 'vol_change']),
    ('11_short_term', ['return', 'rsi', 'macd', 'volatility', 'atr', 'sma_20', 'ema_20']),
    ('12_simple', ['return', 'rsi', 'macd', 'atr', 'sma_20']),
    ('13_return_only', ['return']),
    ('14_rsi_macd', ['rsi', 'macd', 'macd_signal', 'macd_hist']),
    ('15_sma_ema', ['sma_20', 'ema_20', 'sma_60', 'vol_ma_ratio', 'return']),
    ('16_bollinger_atr', ['bollinger_upper', 'bollinger_lower', 'atr', 'volatility', 'vol_change']),
]


def run_experiment(name, features):
    """Run one experiment and parse key metrics from output."""
    features_str = ','.join(features)
    cmd = BASE_CMD + ['--features', features_str, '--output', f'output/feat_exp_{name}']

    print(f"\n{'=' * 70}")
    print(f"Experiment: {name}  ({len(features)} features)")
    print(f"  features={features_str}")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='d:/myStateScoreSys/RL-GHMM')
    elapsed = time.time() - t0
    lines = result.stdout.split('\n') + result.stderr.split('\n')

    metrics = {'name': name, 'n_features': len(features), 'time': elapsed, 'error': None}

    for line in lines:
        line_s = line.strip()
        # Val   Acc: 52.05%  Expected Return: -14.26
        if line_s.startswith('Val') and 'Acc:' in line_s:
            parts = line_s.split()
            for i, p in enumerate(parts):
                if p == 'Acc:':
                    metrics['val_acc'] = float(parts[i+1].rstrip('%'))
                if p == 'Return:':
                    metrics['val_er'] = float(parts[i+1])
        # Best: threshold=0.53, trades=1453, acc=55.33%, exp_ret=2.37
        if line_s.startswith('Best:') and 'threshold=' in line_s:
            # The first element is 'Best: threshold=0.53', strip the 'Best: ' prefix
            clean = line_s.replace('Best:', '', 1).strip()
            for part in clean.split(','):
                part = part.strip()
                if part.startswith('threshold='):
                    metrics['calib_thresh'] = float(part.split('=')[1])
                elif part.startswith('trades='):
                    metrics['calib_trades'] = int(part.split('=')[1])
                elif part.startswith('acc='):
                    metrics['calib_acc'] = float(part.split('=')[1].rstrip('%'))
                elif part.startswith('exp_ret='):
                    metrics['calib_er'] = float(part.split('=')[1])
        if 'ERROR' in line_s or 'Error' in line_s:
            metrics['error'] = line_s

    def fmt_val(v, fmt='.2f', na='N/A'):
        if v is None:
            return na
        return f'{v:{fmt}}'

    print(f"  Time: {elapsed:.1f}s")
    if 'val_acc' in metrics:
        print(f"  Val Acc: {metrics['val_acc']:.2f}%")
    if 'calib_er' in metrics:
        print(f"  Calibrated: thresh={fmt_val(metrics.get('calib_thresh'))}, "
              f"trades={fmt_val(metrics.get('calib_trades'), 'd')}, "
              f"acc={fmt_val(metrics.get('calib_acc'))}%, "
              f"exp_ret={fmt_val(metrics.get('calib_er'))}")

    return metrics


def main():
    # Clean old experiment models to avoid confusion
    old_files = [f for f in os.listdir('output') if f.startswith('feat_exp_') and f.endswith('.pkl')]
    for f in old_files:
        os.remove(os.path.join('output', f))

    results = []
    for name, features in EXPERIMENTS:
        m = run_experiment(name, features)
        results.append(m)

    # Summary table
    print("\n\n" + "=" * 70)
    print("FEATURE EXPERIMENT RESULTS")
    print("=" * 70)
    header = f"{'Name':>18s} | {'#Feat':>5s} | {'Val Acc':>7s} | {'Cal Acc':>7s} | {'Thresh':>6s} | {'Trades':>6s} | {'ExpRet':>8s} | {'Sec':>5s}"
    print(header)
    print("-" * 78)

    # Sort by calibrated expected return (descending)
    def sort_key(m):
        return m.get('calib_er', -999)

    results.sort(key=sort_key, reverse=True)

    for m in results:
        name = m['name']
        nf = m['n_features']
        va = f"{m.get('val_acc', 0):.2f}%" if 'val_acc' in m else '  N/A'
        ca = f"{m.get('calib_acc', 0):.2f}%" if 'calib_acc' in m else '  N/A'
        th = f"{m.get('calib_thresh', 0):.2f}" if 'calib_thresh' in m else '  N/A'
        tr = f"{m.get('calib_trades', 0)}" if 'calib_trades' in m else '  N/A'
        er = f"{m.get('calib_er', 0):.2f}" if 'calib_er' in m else '  N/A'
        tm = f"{m.get('time', 0):.0f}"

        err_flag = " ERROR" if m.get('error') else ""
        print(f"{name:>18s} | {nf:>5d} | {va:>7s} | {ca:>7s} | {th:>6s} | {tr:>6s} | {er:>8s} | {tm:>5s}{err_flag}")

    print("-" * 78)

    # Plot or save detailed results
    with open('output/feature_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to output/feature_experiment_results.json")

    best = results[0] if results else None
    if best and 'calib_er' in best:
        print(f"\n  Best: {best['name']}  (exp_ret={best['calib_er']:.2f}, acc={best.get('calib_acc',0):.2f}%, "
              f"trades={best.get('calib_trades',0)})")
        print(f"  Features: {[f for n, f in EXPERIMENTS if n == best['name']][0]}")


if __name__ == '__main__':
    main()
