"""
Reward Function Evaluator
==========================
Comprehensive evaluation framework for comparing GPRO reward functions.

This module:
1. Loads all reward functions (R1-R5)
2. Runs them on edge case test suite
3. Compares reward signals across functions
4. Generates detailed metrics and statistics
5. Exports results for visualization and thesis analysis

For Master Thesis: Comparative analysis of RL reward functions for radiology grounding.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path to import reward functions
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import reward functions
from Reward_Functions import R1, R2, R3, R4, R5

# Import test suite
from evaluation.edge_case_test_suite import EdgeCaseTestSuite


class RewardFunctionEvaluator:
    """Evaluator for comparing multiple reward functions."""

    def __init__(self):
        """Initialize evaluator with all reward functions."""
        self.reward_functions = {
            'R1': {
                'module': R1,
                'name': 'AP@0.5 Baseline',
                'description': 'Average Precision at IoU 0.5 threshold',
                'type': 'Average Precision',
                'matching': 'Greedy (sorted)',
                'complexity': 'Simple (1 hyperparameter)'
            },
            'R2': {
                'module': R2,
                'name': 'F-beta × IoU Baseline',
                'description': 'F-beta score multiplied by individual IoU scores',
                'type': 'F-beta × IoU',
                'matching': 'Greedy',
                'complexity': 'Simple (3 hyperparameters)'
            },
            'R3': {
                'module': R3,
                'name': 'F-beta × mean_IoU',
                'description': 'F-beta score multiplied by mean IoU of matches',
                'type': 'F-beta × mean_IoU',
                'matching': 'Greedy (sorted)',
                'complexity': 'Simple (3 hyperparameters)'
            },
            'R4': {
                'module': R4,
                'name': 'Enhanced Smooth Spline',
                'description': 'F-beta with cubic spline IoU transformation',
                'type': 'F-beta × smooth_quality',
                'matching': 'Greedy (sorted)',
                'complexity': 'Moderate (4-5 hyperparameters)'
            },
            'R5': {
                'module': R5,
                'name': 'Strict Medical Grounding',
                'description': 'Piecewise quality function with Hungarian matching',
                'type': 'F-beta × piecewise_quality',
                'matching': 'Hungarian (optimal)',
                'complexity': 'Complex (5 hyperparameters)'
            }
        }

        self.test_suite = EdgeCaseTestSuite()
        self.results = None

    def evaluate_single_case(
        self,
        test_case: Dict[str, Any],
        reward_func_module
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case with one reward function.

        Args:
            test_case: Test case dictionary
            reward_func_module: Reward function module (R1, R2, etc.)

        Returns:
            Dictionary with reward and detailed metrics
        """
        try:
            # Call compute_score with return_details=True
            result = reward_func_module.compute_score(
                data_source="edge_case_test",
                solution_str=test_case['prediction'],
                ground_truth=test_case['ground_truth'],
                return_details=True
            )

            # Extract key metrics
            if isinstance(result, dict):
                return {
                    'reward': result.get('reward', result.get('ap', result.get('score', 0.0))),
                    'details': result,
                    'success': True,
                    'error': None
                }
            else:
                # If no details, just return the scalar reward
                return {
                    'reward': float(result),
                    'details': {'reward': float(result)},
                    'success': True,
                    'error': None
                }
        except Exception as e:
            return {
                'reward': 0.0,
                'details': {},
                'success': False,
                'error': str(e)
            }

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all reward functions on all test cases.

        Returns:
            DataFrame with results
        """
        results_list = []

        test_cases = self.test_suite.get_all_tests()
        total_tests = len(test_cases)

        print(f"Evaluating {len(self.reward_functions)} reward functions on {total_tests} test cases...")
        print("=" * 80)

        for rf_name, rf_info in self.reward_functions.items():
            print(f"\nEvaluating {rf_name}: {rf_info['name']}")
            print("-" * 80)

            rf_module = rf_info['module']

            for i, test_case in enumerate(test_cases, 1):
                # Progress indicator
                if i % 10 == 0:
                    print(f"  Progress: {i}/{total_tests} tests completed")

                # Evaluate test case
                result = self.evaluate_single_case(test_case, rf_module)

                # Store result
                results_list.append({
                    'reward_function': rf_name,
                    'rf_full_name': rf_info['name'],
                    'rf_type': rf_info['type'],
                    'rf_matching': rf_info['matching'],
                    'rf_complexity': rf_info['complexity'],
                    'test_id': test_case['id'],
                    'test_name': test_case['name'],
                    'category': test_case['category'],
                    'edge_case_type': test_case.get('edge_case_type', 'unknown'),
                    'clinical_scenario': test_case['clinical_scenario'],
                    'reward': result['reward'],
                    'success': result['success'],
                    'error': result['error'],
                    'prediction': test_case['prediction'],
                    'ground_truth': test_case['ground_truth'],
                    'details': json.dumps(result['details'])
                })

        # Convert to DataFrame
        self.results = pd.DataFrame(results_list)

        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print(f"Total evaluations: {len(self.results)}")
        print(f"Successful: {self.results['success'].sum()}")
        print(f"Failed: {(~self.results['success']).sum()}")

        return self.results

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics across all reward functions.

        Returns:
            Dictionary with statistics
        """
        if self.results is None:
            raise ValueError("Must run evaluate_all() first")

        stats = {}

        # Overall statistics per reward function
        for rf_name in self.results['reward_function'].unique():
            rf_data = self.results[self.results['reward_function'] == rf_name]

            stats[rf_name] = {
                'overall': {
                    'mean': float(rf_data['reward'].mean()),
                    'std': float(rf_data['reward'].std()),
                    'min': float(rf_data['reward'].min()),
                    'max': float(rf_data['reward'].max()),
                    'median': float(rf_data['reward'].median()),
                    'q25': float(rf_data['reward'].quantile(0.25)),
                    'q75': float(rf_data['reward'].quantile(0.75)),
                    'success_rate': float(rf_data['success'].mean())
                },
                'by_category': {},
                'by_edge_case_type': {}
            }

            # Statistics by category
            for category in rf_data['category'].unique():
                cat_data = rf_data[rf_data['category'] == category]
                stats[rf_name]['by_category'][category] = {
                    'mean': float(cat_data['reward'].mean()),
                    'std': float(cat_data['reward'].std()),
                    'count': int(len(cat_data))
                }

            # Statistics by edge case type
            for edge_type in rf_data['edge_case_type'].unique():
                edge_data = rf_data[rf_data['edge_case_type'] == edge_type]
                stats[rf_name]['by_edge_case_type'][edge_type] = {
                    'mean': float(edge_data['reward'].mean()),
                    'std': float(edge_data['reward'].std()),
                    'count': int(len(edge_data))
                }

        return stats

    def compute_signal_differences(self) -> pd.DataFrame:
        """
        Compute pairwise differences in reward signals between functions.

        Returns:
            DataFrame with signal differences for each test case
        """
        if self.results is None:
            raise ValueError("Must run evaluate_all() first")

        # Pivot to get reward functions as columns
        pivot = self.results.pivot(
            index='test_id',
            columns='reward_function',
            values='reward'
        )

        # Compute all pairwise differences
        differences = []
        rf_names = list(self.reward_functions.keys())

        for i, rf1 in enumerate(rf_names):
            for rf2 in rf_names[i+1:]:
                diff = pivot[rf1] - pivot[rf2]
                differences.append({
                    'comparison': f'{rf1} - {rf2}',
                    'rf1': rf1,
                    'rf2': rf2,
                    'mean_diff': float(diff.mean()),
                    'std_diff': float(diff.std()),
                    'abs_mean_diff': float(diff.abs().mean()),
                    'max_diff': float(diff.max()),
                    'min_diff': float(diff.min()),
                    'correlation': float(pivot[rf1].corr(pivot[rf2]))
                })

        return pd.DataFrame(differences)

    def generate_report(self, output_dir: str = "evaluation/results"):
        """
        Generate comprehensive evaluation report.

        Args:
            output_dir: Directory to save results
        """
        if self.results is None:
            raise ValueError("Must run evaluate_all() first")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        csv_path = output_path / f"evaluation_results_{timestamp}.csv"
        self.results.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Compute and save statistics
        stats = self.compute_statistics()
        stats_path = output_path / f"statistics_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")

        # Compute and save signal differences
        signal_diffs = self.compute_signal_differences()
        diff_path = output_path / f"signal_differences_{timestamp}.csv"
        signal_diffs.to_csv(diff_path, index=False)
        print(f"Signal differences saved to: {diff_path}")

        # Generate summary report
        self._generate_summary_report(output_path, timestamp, stats, signal_diffs)

        return {
            'results_csv': str(csv_path),
            'statistics_json': str(stats_path),
            'differences_csv': str(diff_path)
        }

    def _generate_summary_report(
        self,
        output_path: Path,
        timestamp: str,
        stats: Dict[str, Any],
        signal_diffs: pd.DataFrame
    ):
        """Generate a human-readable summary report."""
        report_path = output_path / f"summary_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("REWARD FUNCTION EVALUATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test cases: {len(self.test_suite.get_all_tests())}\n")
            f.write(f"Reward functions evaluated: {len(self.reward_functions)}\n")
            f.write("\n")

            # Overall statistics
            f.write("=" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            for rf_name in sorted(stats.keys()):
                rf_info = self.reward_functions[rf_name]
                rf_stats = stats[rf_name]['overall']

                f.write(f"{rf_name}: {rf_info['name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Type: {rf_info['type']}\n")
                f.write(f"  Matching: {rf_info['matching']}\n")
                f.write(f"  Complexity: {rf_info['complexity']}\n")
                f.write(f"  Mean reward: {rf_stats['mean']:.4f} ± {rf_stats['std']:.4f}\n")
                f.write(f"  Median reward: {rf_stats['median']:.4f}\n")
                f.write(f"  Range: [{rf_stats['min']:.4f}, {rf_stats['max']:.4f}]\n")
                f.write(f"  IQR: [{rf_stats['q25']:.4f}, {rf_stats['q75']:.4f}]\n")
                f.write(f"  Success rate: {rf_stats['success_rate']*100:.1f}%\n")
                f.write("\n")

            # Category-wise comparison
            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE BY CATEGORY\n")
            f.write("=" * 80 + "\n\n")

            categories = self.results['category'].unique()
            for category in sorted(categories):
                f.write(f"{category.upper()}\n")
                f.write("-" * 80 + "\n")

                for rf_name in sorted(stats.keys()):
                    cat_stats = stats[rf_name]['by_category'].get(category, {})
                    if cat_stats:
                        f.write(f"  {rf_name:4s}: {cat_stats['mean']:.4f} ± {cat_stats['std']:.4f} (n={cat_stats['count']})\n")
                f.write("\n")

            # Signal differences
            f.write("=" * 80 + "\n")
            f.write("SIGNAL DIFFERENCES (Pairwise Comparisons)\n")
            f.write("=" * 80 + "\n\n")

            for _, row in signal_diffs.iterrows():
                f.write(f"{row['comparison']}:\n")
                f.write(f"  Mean difference: {row['mean_diff']:+.4f}\n")
                f.write(f"  Abs mean difference: {row['abs_mean_diff']:.4f}\n")
                f.write(f"  Range: [{row['min_diff']:+.4f}, {row['max_diff']:+.4f}]\n")
                f.write(f"  Correlation: {row['correlation']:.4f}\n")
                f.write("\n")

        print(f"Summary report saved to: {report_path}")


def main():
    """Run complete evaluation."""
    print("=" * 80)
    print("REWARD FUNCTION EVALUATION")
    print("=" * 80)

    # Create evaluator
    evaluator = RewardFunctionEvaluator()

    # Run evaluation
    results_df = evaluator.evaluate_all()

    # Generate report
    file_paths = evaluator.generate_report()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    for key, path in file_paths.items():
        print(f"  {key}: {path}")

    return evaluator


if __name__ == "__main__":
    evaluator = main()
