"""
Plot training results from GRSN experiments.

Usage:
    python scripts/plot_results.py --env Pendulum-V-v0 --models rnn,snn --seeds 0,1,2,3,4
"""

import argparse
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_results(result_dir, pattern="*.pth"):
    """Load all result files matching the pattern."""
    result_files = glob.glob(os.path.join(result_dir, pattern))
    results = {}
    for f in result_files:
        name = os.path.basename(f).replace(".pth", "")
        try:
            data = torch.load(f)
            results[name] = data
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


def smooth_curve(x, y, window=5):
    """Smooth a curve using moving average."""
    if len(y) < window:
        return x, y
    weights = np.ones(window) / window
    y_smooth = np.convolve(y, weights, mode='valid')
    x_smooth = x[window-1:]
    return x_smooth, y_smooth


def plot_learning_curves(results, output_path=None, smooth_window=5):
    """Plot learning curves for all results."""
    plt.figure(figsize=(12, 6))

    for name, data in sorted(results.items()):
        x = np.array(data['x'])
        y = np.array(data['y'])

        if smooth_window > 0:
            x, y = smooth_curve(x, y, smooth_window)

        plt.plot(x, y, label=name, alpha=0.8)

    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Learning Curves', fontsize=14)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def aggregate_by_model(results, model_types):
    """Aggregate results by model type and compute mean/std."""
    aggregated = {mt: [] for mt in model_types}

    for name, data in results.items():
        for mt in model_types:
            if mt in name.lower():
                aggregated[mt].append((data['x'], data['y']))
                break

    # Compute mean and std for each model type
    summary = {}
    for mt, curves in aggregated.items():
        if not curves:
            continue

        # Interpolate to common x values
        max_len = max(len(c[0]) for c in curves)
        common_x = curves[0][0][:max_len]

        all_y = []
        for x, y in curves:
            if len(y) >= max_len:
                all_y.append(y[:max_len])

        if all_y:
            y_array = np.array(all_y)
            summary[mt] = {
                'x': common_x,
                'mean': y_array.mean(axis=0),
                'std': y_array.std(axis=0),
                'min': y_array.min(axis=0),
                'max': y_array.max(axis=0),
                'n': len(all_y)
            }

    return summary


def plot_aggregated(summary, output_path=None, smooth_window=5):
    """Plot aggregated learning curves with confidence intervals."""
    plt.figure(figsize=(12, 6))

    for model_type, data in sorted(summary.items()):
        x = np.array(data['x'])
        mean = data['mean']
        std = data['std']
        n = data['n']

        if smooth_window > 0:
            x, mean = smooth_curve(x, mean, smooth_window)
            _, std = smooth_curve(data['x'], std, smooth_window)

        plt.plot(x, mean, label=f"{model_type} (n={n})", linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Aggregated Learning Curves (Mean ± Std)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot GRSN experiment results')
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Smoothing window size (0 for no smoothing)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate results by model type')
    parser.add_argument('--models', type=str, default='rnn,snn,lif,recurrentlif,grsnwo',
                        help='Comma-separated list of model types for aggregation')

    args = parser.parse_args()

    # Load results
    result_dir = os.path.join(args.result_dir, args.env)
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return

    print(f"Loading results from: {result_dir}")
    results = load_results(result_dir)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results to plot!")
        return

    # Plot
    if args.aggregate:
        model_types = [m.strip() for m in args.models.split(',')]
        summary = aggregate_by_model(results, model_types)

        if not summary:
            print("No data to aggregate!")
            return

        print(f"\nAggregated data for {len(summary)} model types:")
        for mt, data in summary.items():
            print(f"  {mt}: {data['n']} runs")

        output_path = args.output or os.path.join(result_dir, 'aggregated.png')
        plot_aggregated(summary, output_path, args.smooth)
    else:
        output_path = args.output or os.path.join(result_dir, 'learning_curves.png')
        plot_learning_curves(results, output_path, args.smooth)

    print("\nDone!")


if __name__ == "__main__":
    main()
