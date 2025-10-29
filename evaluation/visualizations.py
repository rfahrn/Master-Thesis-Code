"""
Visualization Module for Reward Function Analysis
==================================================
Generate publication-quality plots for Master Thesis.

This module creates:
1. Reward signal comparison plots
2. Category-wise performance heatmaps
3. IoU-Reward relationship curves
4. Signal difference distributions
5. Edge case performance radar charts
6. Correlation matrices

For Master Thesis: Comparative analysis of RL reward functions for radiology grounding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class RewardFunctionVisualizer:
    """Create visualizations for reward function analysis."""

    def __init__(self, results_csv: str, stats_json: Optional[str] = None):
        """
        Initialize visualizer with evaluation results.

        Args:
            results_csv: Path to evaluation results CSV
            stats_json: Path to statistics JSON (optional)
        """
        self.results = pd.read_csv(results_csv)
        self.stats = None

        if stats_json and Path(stats_json).exists():
            with open(stats_json, 'r') as f:
                self.stats = json.load(f)

        self.reward_functions = sorted(self.results['reward_function'].unique())
        self.categories = sorted(self.results['category'].unique())

    def plot_reward_distributions(self, output_path: str = "evaluation/plots/reward_distributions.png"):
        """
        Plot reward distributions for each reward function.

        Creates violin plots showing the distribution of rewards.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Overall distribution
        ax = axes[0]
        sns.violinplot(
            data=self.results,
            x='reward_function',
            y='reward',
            ax=ax,
            inner='box'
        )
        ax.set_title('Reward Distribution Across All Test Cases', fontweight='bold')
        ax.set_xlabel('Reward Function')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

        # Add mean markers
        for i, rf in enumerate(self.reward_functions):
            mean_val = self.results[self.results['reward_function'] == rf]['reward'].mean()
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else '')

        ax.legend()

        # Plot 2: Distribution by category
        ax = axes[1]
        sns.boxplot(
            data=self.results,
            x='category',
            y='reward',
            hue='reward_function',
            ax=ax
        )
        ax.set_title('Reward Distribution by Test Category', fontweight='bold')
        ax.set_xlabel('Test Category')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='Reward Function', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_category_heatmap(self, output_path: str = "evaluation/plots/category_heatmap.png"):
        """
        Create heatmap showing mean rewards per category per function.
        """
        # Compute mean rewards per category per function
        pivot_data = self.results.groupby(['category', 'reward_function'])['reward'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='category', columns='reward_function', values='reward')

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Mean Reward'},
            ax=ax
        )

        ax.set_title('Mean Reward by Category and Function', fontweight='bold', pad=20)
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Test Category', fontweight='bold')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_edge_case_comparison(self, output_path: str = "evaluation/plots/edge_case_comparison.png"):
        """
        Compare reward functions on specific edge case types.
        """
        # Get edge case types
        edge_types = self.results['edge_case_type'].unique()

        # Compute mean rewards per edge case type
        edge_case_data = self.results.groupby(['edge_case_type', 'reward_function'])['reward'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar chart
        x = np.arange(len(edge_types))
        width = 0.15
        offset = width * (len(self.reward_functions) - 1) / 2

        for i, rf in enumerate(self.reward_functions):
            rf_data = edge_case_data[edge_case_data['reward_function'] == rf]
            values = [rf_data[rf_data['edge_case_type'] == et]['reward'].values[0]
                     if len(rf_data[rf_data['edge_case_type'] == et]) > 0 else 0
                     for et in edge_types]

            ax.bar(x + (i - len(self.reward_functions)//2) * width, values, width, label=rf)

        ax.set_xlabel('Edge Case Type', fontweight='bold')
        ax.set_ylabel('Mean Reward', fontweight='bold')
        ax.set_title('Mean Reward by Edge Case Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(edge_types, rotation=45, ha='right')
        ax.legend(title='Reward Function')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_signal_correlation(self, output_path: str = "evaluation/plots/signal_correlation.png"):
        """
        Create correlation matrix of reward signals between functions.
        """
        # Pivot to get reward functions as columns
        pivot = self.results.pivot(
            index='test_id',
            columns='reward_function',
            values='reward'
        )

        # Compute correlation matrix
        corr_matrix = pivot.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0.9,
            vmin=0.5,
            vmax=1.0,
            square=True,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )

        ax.set_title('Reward Signal Correlation Matrix', fontweight='bold', pad=20)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_pairwise_scatter(self, rf1: str, rf2: str, output_path: Optional[str] = None):
        """
        Create scatter plot comparing two reward functions.

        Args:
            rf1: First reward function (e.g., 'R1')
            rf2: Second reward function (e.g., 'R4')
            output_path: Output path (default: auto-generated)
        """
        if output_path is None:
            output_path = f"evaluation/plots/scatter_{rf1}_vs_{rf2}.png"

        # Pivot to get both functions
        pivot = self.results.pivot(
            index='test_id',
            columns='reward_function',
            values='reward'
        )

        # Merge with category info
        test_categories = self.results[['test_id', 'category']].drop_duplicates()
        plot_data = pivot.reset_index().merge(test_categories, on='test_id')

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot with category colors
        for category in self.categories:
            cat_data = plot_data[plot_data['category'] == category]
            ax.scatter(
                cat_data[rf1],
                cat_data[rf2],
                label=category,
                alpha=0.6,
                s=50
            )

        # Add diagonal line (x=y)
        min_val = min(pivot[rf1].min(), pivot[rf2].min())
        max_val = max(pivot[rf1].max(), pivot[rf2].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

        # Compute correlation
        correlation = pivot[rf1].corr(pivot[rf2])

        ax.set_xlabel(f'{rf1} Reward', fontweight='bold')
        ax.set_ylabel(f'{rf2} Reward', fontweight='bold')
        ax.set_title(f'Reward Comparison: {rf1} vs {rf2}\n(Correlation: {correlation:.3f})',
                    fontweight='bold')
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_iou_reward_curves(self, output_path: str = "evaluation/plots/iou_reward_curves.png"):
        """
        Plot theoretical IoU-Reward curves for each function.

        This visualizes how each reward function transforms IoU values.
        """
        # Generate IoU values
        iou_values = np.linspace(0, 1, 100)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: All functions together
        ax = axes[0, 0]
        for rf in self.reward_functions:
            # Get one-to-one test cases for this function
            one_to_one = self.results[
                (self.results['reward_function'] == rf) &
                (self.results['edge_case_type'] == 'one_to_one')
            ]

            if len(one_to_one) > 0:
                # Sort by expected IoU if available, else by reward
                if 'expected_iou' in self.results.columns:
                    one_to_one = one_to_one.dropna(subset=['expected_iou'])
                    one_to_one = one_to_one.sort_values('expected_iou')
                    ax.plot(one_to_one['expected_iou'], one_to_one['reward'],
                           'o-', label=rf, alpha=0.7, markersize=5)
                else:
                    one_to_one = one_to_one.sort_values('reward')
                    ax.plot(one_to_one['reward'], label=rf, alpha=0.7)

        ax.set_xlabel('IoU', fontweight='bold')
        ax.set_ylabel('Reward', fontweight='bold')
        ax.set_title('IoU-Reward Relationship (All Functions)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: R1 (AP@0.5) - step function behavior
        ax = axes[0, 1]
        r1_data = self.results[
            (self.results['reward_function'] == 'R1') &
            (self.results['edge_case_type'] == 'one_to_one')
        ].copy()
        if len(r1_data) > 0:
            r1_data = r1_data.sort_values('reward')
            ax.scatter(range(len(r1_data)), r1_data['reward'], alpha=0.6)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_xlabel('Test Case Index (sorted by reward)', fontweight='bold')
            ax.set_ylabel('Reward', fontweight='bold')
            ax.set_title('R1 (AP@0.5): Threshold Behavior', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: R4 (Smooth spline) - smooth curve
        ax = axes[1, 0]
        r4_data = self.results[
            (self.results['reward_function'] == 'R4') &
            (self.results['edge_case_type'] == 'one_to_one')
        ].copy()
        if len(r4_data) > 0:
            r4_data = r4_data.sort_values('reward')
            ax.scatter(range(len(r4_data)), r4_data['reward'], alpha=0.6, color='orange')
            ax.set_xlabel('Test Case Index (sorted by reward)', fontweight='bold')
            ax.set_ylabel('Reward', fontweight='bold')
            ax.set_title('R4 (Smooth Spline): Gradient Behavior', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Plot 4: Reward variance comparison
        ax = axes[1, 1]
        variance_data = self.results.groupby('reward_function')['reward'].agg(['std', 'var']).reset_index()
        ax.bar(variance_data['reward_function'], variance_data['std'])
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontweight='bold')
        ax.set_title('Reward Signal Variability', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def plot_summary_statistics(self, output_path: str = "evaluation/plots/summary_statistics.png"):
        """
        Create summary statistics visualization.
        """
        if self.stats is None:
            print("Warning: No statistics JSON provided, computing from results...")
            # Compute basic stats
            stats_data = []
            for rf in self.reward_functions:
                rf_data = self.results[self.results['reward_function'] == rf]['reward']
                stats_data.append({
                    'reward_function': rf,
                    'mean': rf_data.mean(),
                    'median': rf_data.median(),
                    'std': rf_data.std(),
                    'min': rf_data.min(),
                    'max': rf_data.max()
                })
            stats_df = pd.DataFrame(stats_data)
        else:
            # Extract from loaded stats
            stats_data = []
            for rf in self.reward_functions:
                overall = self.stats[rf]['overall']
                stats_data.append({
                    'reward_function': rf,
                    'mean': overall['mean'],
                    'median': overall['median'],
                    'std': overall['std'],
                    'min': overall['min'],
                    'max': overall['max']
                })
            stats_df = pd.DataFrame(stats_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Mean rewards with error bars
        ax = axes[0, 0]
        ax.bar(stats_df['reward_function'], stats_df['mean'], yerr=stats_df['std'],
              capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Mean Reward', fontweight='bold')
        ax.set_title('Mean Reward ± Std Dev', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Median rewards
        ax = axes[0, 1]
        ax.bar(stats_df['reward_function'], stats_df['median'],
              alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Median Reward', fontweight='bold')
        ax.set_title('Median Reward', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Range (min to max)
        ax = axes[1, 0]
        for i, rf in enumerate(stats_df['reward_function']):
            ax.plot([i, i], [stats_df.loc[i, 'min'], stats_df.loc[i, 'max']],
                   'o-', linewidth=3, markersize=8)
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(stats_df['reward_function'])
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Reward Range', fontweight='bold')
        ax.set_title('Reward Range (Min to Max)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Coefficient of variation (std/mean)
        ax = axes[1, 1]
        cv = stats_df['std'] / stats_df['mean']
        ax.bar(stats_df['reward_function'], cv, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Reward Function', fontweight='bold')
        ax.set_ylabel('Coefficient of Variation', fontweight='bold')
        ax.set_title('Relative Variability (CV = σ/μ)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    def generate_all_plots(self, output_dir: str = "evaluation/plots"):
        """
        Generate all visualization plots.

        Args:
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        # Generate all plots
        self.plot_reward_distributions(f"{output_dir}/reward_distributions.png")
        self.plot_category_heatmap(f"{output_dir}/category_heatmap.png")
        self.plot_edge_case_comparison(f"{output_dir}/edge_case_comparison.png")
        self.plot_signal_correlation(f"{output_dir}/signal_correlation.png")
        self.plot_iou_reward_curves(f"{output_dir}/iou_reward_curves.png")
        self.plot_summary_statistics(f"{output_dir}/summary_statistics.png")

        # Generate pairwise scatter plots for key comparisons
        key_comparisons = [
            ('R1', 'R3'),  # Baseline vs Production
            ('R3', 'R4'),  # Production vs Enhanced
            ('R4', 'R5'),  # Enhanced vs Strict
            ('R1', 'R5'),  # Baseline vs Strict
        ]

        for rf1, rf2 in key_comparisons:
            self.plot_pairwise_scatter(rf1, rf2, f"{output_dir}/scatter_{rf1}_vs_{rf2}.png")

        print("\n" + "=" * 80)
        print("ALL PLOTS GENERATED")
        print("=" * 80)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate reward function visualizations')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results CSV')
    parser.add_argument('--stats', type=str, default=None,
                       help='Path to statistics JSON (optional)')
    parser.add_argument('--output', type=str, default='evaluation/plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    visualizer = RewardFunctionVisualizer(args.results, args.stats)
    visualizer.generate_all_plots(args.output)


if __name__ == "__main__":
    main()
