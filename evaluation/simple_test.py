"""
Simple Test Script for Reward Functions
========================================
Runs basic evaluation without requiring matplotlib/pandas/seaborn.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import reward functions
from Reward_Functions import R1, R2, R3, R4, R5

# Import test suite
from evaluation.edge_case_test_suite import EdgeCaseTestSuite


def run_simple_test():
    """Run simple evaluation on a few test cases."""

    print("=" * 80)
    print("SIMPLE REWARD FUNCTION TEST")
    print("=" * 80)

    # Create test suite
    suite = EdgeCaseTestSuite()

    # Select a few representative test cases
    test_ids = [
        'basic_001',  # True negative
        'basic_002',  # Perfect match
        'basic_003',  # Hallucination
        'basic_004',  # Missed detection
        'iou_003',    # Just-above-threshold
        'multi_001',  # Two perfect matches
    ]

    reward_functions = {
        'R1': R1,
        'R2': R2,
        'R3': R3,
        'R4': R4,
        'R5': R5
    }

    print(f"\nTesting {len(test_ids)} edge cases with {len(reward_functions)} reward functions")
    print("=" * 80)

    results = []

    for test_id in test_ids:
        test = suite.get_test_by_id(test_id)

        print(f"\n[{test_id}] {test['name']}")
        print("-" * 80)
        print(f"Description: {test['description']}")
        print(f"Category: {test['category']}")
        print(f"Edge case type: {test['edge_case_type']}")
        print(f"\nRewards:")

        test_results = {'test_id': test_id, 'test_name': test['name']}

        for rf_name, rf_module in reward_functions.items():
            try:
                # Call compute_score
                reward = rf_module.compute_score(
                    data_source="test",
                    solution_str=test['prediction'],
                    ground_truth=test['ground_truth']
                )

                # Handle case where detailed dict is returned
                if isinstance(reward, dict):
                    reward = reward.get('reward', reward.get('ap', reward.get('score', 0.0)))

                print(f"  {rf_name}: {reward:.4f}")
                test_results[rf_name] = reward

            except Exception as e:
                print(f"  {rf_name}: ERROR - {str(e)}")
                test_results[rf_name] = None

        results.append(test_results)

    # Compute simple statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for rf_name in reward_functions.keys():
        rewards = [r[rf_name] for r in results if r[rf_name] is not None]
        if rewards:
            mean = sum(rewards) / len(rewards)
            minimum = min(rewards)
            maximum = max(rewards)
            print(f"\n{rf_name}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Min:  {minimum:.4f}")
            print(f"  Max:  {maximum:.4f}")
            print(f"  Range: [{minimum:.4f}, {maximum:.4f}]")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_simple_test()
