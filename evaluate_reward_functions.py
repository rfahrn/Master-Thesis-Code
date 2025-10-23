#!/usr/bin/env python3
"""
Comprehensive Reward Function Evaluation for Radiology Grounding/Detection
Compares multiple reward functions for GRPO-based VLM optimization

Author: Master Thesis Evaluation Framework
Purpose: Find optimal reward function for radiology grounding with multiple bounding boxes
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import json
from collections import defaultdict

# Add Reward Functions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Reward Functions'))

# Import all reward function modules
import R1  # AP@0.5
import R2  # Continuous IoU
import R3  # F-beta × mean_IoU
import R4  # Enhanced smooth gradients
import R5  # Soft reward with partial credit
import R6  # F1-weighted IoU

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class TestCase:
    """Represents a grounding/detection test scenario"""
    name: str
    description: str
    ground_truth: List[List[float]]
    predictions: List[List[float]]
    scenario_type: str  # e.g., "perfect", "partial", "false_positive", "multi_box"
    expected_behavior: str


@dataclass
class RewardFunctionMetrics:
    """Metrics for a single reward function"""
    name: str
    rewards: List[float] = field(default_factory=list)
    scenario_rewards: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    statistics: Dict[str, float] = field(default_factory=dict)


class RewardFunctionEvaluator:
    """Evaluates and compares multiple reward functions"""

    def __init__(self):
        self.reward_functions = {
            'R3_FBeta_MeanIoU': R3,
            'R4_Smooth_Gradients': R4,
            'R5_Soft_Partial_Credit': R5,
            'R6_F1_Weighted_IoU': R6
        }
        self.test_cases = self._create_test_cases()
        self.results = {}

    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test scenarios for radiology grounding"""

        test_cases = []

        # 1. PERFECT MATCH - Single box
        test_cases.append(TestCase(
            name="Perfect_Single_Box",
            description="Perfect match for single finding",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[[100, 100, 200, 200]],
            scenario_type="perfect",
            expected_behavior="Maximum reward (IoU=1.0)"
        ))

        # 2. PERFECT MATCH - Multiple boxes
        test_cases.append(TestCase(
            name="Perfect_Multi_Box",
            description="Perfect match for 3 findings",
            ground_truth=[[100, 100, 200, 200], [300, 300, 400, 400], [500, 100, 600, 200]],
            predictions=[[100, 100, 200, 200], [300, 300, 400, 400], [500, 100, 600, 200]],
            scenario_type="perfect",
            expected_behavior="Maximum reward for multi-box"
        ))

        # 3. HIGH OVERLAP - Good localization (IoU~0.8)
        test_cases.append(TestCase(
            name="High_Overlap_Single",
            description="Good but not perfect localization (IoU≈0.8)",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[[110, 110, 200, 200]],
            scenario_type="partial",
            expected_behavior="High reward, should encourage further refinement"
        ))

        # 4. MODERATE OVERLAP - Acceptable localization (IoU~0.6)
        test_cases.append(TestCase(
            name="Moderate_Overlap",
            description="Acceptable localization (IoU≈0.6)",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[[120, 120, 200, 200]],
            scenario_type="partial",
            expected_behavior="Moderate reward, clear improvement needed"
        ))

        # 5. LOW OVERLAP - Poor but not useless (IoU~0.3)
        test_cases.append(TestCase(
            name="Low_Overlap",
            description="Poor localization (IoU≈0.3)",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[[150, 150, 250, 250]],
            scenario_type="partial",
            expected_behavior="Low reward, but some partial credit?"
        ))

        # 6. BARELY OVERLAP - Marginal detection (IoU~0.1-0.2)
        test_cases.append(TestCase(
            name="Barely_Overlap",
            description="Barely overlapping (IoU≈0.15)",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[[180, 180, 280, 280]],
            scenario_type="partial",
            expected_behavior="Very low or zero reward"
        ))

        # 7. FALSE POSITIVE - Hallucination
        test_cases.append(TestCase(
            name="False_Positive_Single",
            description="Model hallucinates finding",
            ground_truth=[],
            predictions=[[100, 100, 200, 200]],
            scenario_type="false_positive",
            expected_behavior="Negative reward penalty"
        ))

        # 8. FALSE NEGATIVES - Missed detection
        test_cases.append(TestCase(
            name="False_Negative_Single",
            description="Model misses finding",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[],
            scenario_type="false_negative",
            expected_behavior="Negative reward penalty"
        ))

        # 9. TRUE NEGATIVE - Correctly identifies no findings
        test_cases.append(TestCase(
            name="True_Negative",
            description="Correctly no boxes when no findings",
            ground_truth=[],
            predictions=[],
            scenario_type="true_negative",
            expected_behavior="Small positive reward (NO_BOX_BONUS)"
        ))

        # 10. MULTI-BOX: Some correct, some missed
        test_cases.append(TestCase(
            name="Multi_Box_Partial_Recall",
            description="2/3 findings detected (missed one)",
            ground_truth=[[100, 100, 200, 200], [300, 300, 400, 400], [500, 100, 600, 200]],
            predictions=[[100, 100, 200, 200], [300, 300, 400, 400]],
            scenario_type="multi_box_partial",
            expected_behavior="Partial reward, penalized for miss"
        ))

        # 11. MULTI-BOX: All found but some imprecise
        test_cases.append(TestCase(
            name="Multi_Box_Imprecise",
            description="All 3 found but varying IoU (1.0, 0.7, 0.4)",
            ground_truth=[[100, 100, 200, 200], [300, 300, 400, 400], [500, 100, 600, 200]],
            predictions=[[100, 100, 200, 200], [310, 310, 400, 400], [520, 120, 600, 200]],
            scenario_type="multi_box_varying",
            expected_behavior="Good recall, but penalized for poor localization"
        ))

        # 12. MULTI-BOX: Extra false positive
        test_cases.append(TestCase(
            name="Multi_Box_Extra_FP",
            description="2 correct + 1 hallucinated",
            ground_truth=[[100, 100, 200, 200], [300, 300, 400, 400]],
            predictions=[[100, 100, 200, 200], [300, 300, 400, 400], [500, 100, 600, 200]],
            scenario_type="multi_box_fp",
            expected_behavior="Penalized for false positive"
        ))

        # 13. CHALLENGING: Many boxes with mixed quality
        test_cases.append(TestCase(
            name="Complex_Multi_Box",
            description="5 GT boxes, 4 predictions with varying IoU",
            ground_truth=[
                [50, 50, 150, 150],    # Will match prediction 1 (perfect)
                [200, 200, 300, 300],  # Will match prediction 2 (high IoU)
                [400, 100, 500, 200],  # Will match prediction 3 (moderate IoU)
                [100, 400, 200, 500],  # Will match prediction 4 (low IoU)
                [600, 600, 700, 700]   # Will be missed (FN)
            ],
            predictions=[
                [50, 50, 150, 150],     # Perfect match (IoU=1.0)
                [205, 205, 300, 300],   # High overlap (IoU≈0.8)
                [420, 120, 500, 200],   # Moderate overlap (IoU≈0.5)
                [130, 430, 200, 500],   # Low overlap (IoU≈0.3)
            ],
            scenario_type="complex_multi",
            expected_behavior="Balanced reward considering all factors"
        ))

        # 14. STRESS TEST: Many false positives
        test_cases.append(TestCase(
            name="Many_False_Positives",
            description="1 GT box but model predicts 5",
            ground_truth=[[100, 100, 200, 200]],
            predictions=[
                [100, 100, 200, 200],  # 1 correct
                [300, 300, 400, 400],  # FP
                [500, 100, 600, 200],  # FP
                [100, 400, 200, 500],  # FP
                [600, 600, 700, 700]   # FP
            ],
            scenario_type="many_fp",
            expected_behavior="Heavily penalized for hallucinations"
        ))

        # 15. STRESS TEST: Many false negatives
        test_cases.append(TestCase(
            name="Many_False_Negatives",
            description="5 GT boxes but model predicts only 1",
            ground_truth=[
                [50, 50, 150, 150],
                [200, 200, 300, 300],
                [400, 100, 500, 200],
                [100, 400, 200, 500],
                [600, 600, 700, 700]
            ],
            predictions=[[50, 50, 150, 150]],  # Only 1 correct
            scenario_type="many_fn",
            expected_behavior="Heavily penalized for missed findings"
        ))

        return test_cases

    def _format_boxes_for_reward(self, boxes: List[List[float]]) -> str:
        """Format bounding boxes as expected by reward functions"""
        if not boxes:
            return "no box"

        formatted = []
        for box in boxes:
            formatted.append(f"[{box[0]},{box[1]},{box[2]},{box[3]}]")
        return " ".join(formatted)

    def evaluate_all(self) -> Dict[str, RewardFunctionMetrics]:
        """Evaluate all reward functions on all test cases"""

        print("=" * 80)
        print("REWARD FUNCTION EVALUATION FOR RADIOLOGY GROUNDING")
        print("=" * 80)
        print(f"\nEvaluating {len(self.reward_functions)} reward functions")
        print(f"across {len(self.test_cases)} test scenarios\n")

        # Initialize metrics
        for rf_name in self.reward_functions.keys():
            self.results[rf_name] = RewardFunctionMetrics(name=rf_name)

        # Evaluate each test case
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] {test_case.name}")
            print(f"    Description: {test_case.description}")
            print(f"    GT boxes: {len(test_case.ground_truth)}, Pred boxes: {len(test_case.predictions)}")

            # Format inputs
            gt_str = self._format_boxes_for_reward(test_case.ground_truth)
            pred_str = self._format_boxes_for_reward(test_case.predictions)

            # Evaluate each reward function
            for rf_name, rf_module in self.reward_functions.items():
                try:
                    reward = rf_module.compute_score(
                        data_source="evaluation",
                        solution_str=pred_str,
                        ground_truth=gt_str,
                        extra_info={},
                        return_details=False
                    )

                    self.results[rf_name].rewards.append(reward)
                    self.results[rf_name].scenario_rewards[test_case.scenario_type].append(reward)

                    print(f"    {rf_name}: {reward:.4f}")

                except Exception as e:
                    print(f"    {rf_name}: ERROR - {str(e)}")
                    self.results[rf_name].rewards.append(np.nan)

        # Compute statistics
        self._compute_statistics()

        return self.results

    def _compute_statistics(self):
        """Compute summary statistics for each reward function"""

        for rf_name, metrics in self.results.items():
            rewards = np.array([r for r in metrics.rewards if not np.isnan(r)])

            if len(rewards) > 0:
                metrics.statistics = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'median': np.median(rewards),
                    'range': np.max(rewards) - np.min(rewards),
                    'q25': np.percentile(rewards, 25),
                    'q75': np.percentile(rewards, 75)
                }

    def plot_results(self, output_dir: str = "."):
        """Generate comprehensive visualization plots"""

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        # 1. Overall reward distribution comparison
        self._plot_reward_distributions(output_dir)

        # 2. Scenario-wise comparison
        self._plot_scenario_comparison(output_dir)

        # 3. Test case heatmap
        self._plot_heatmap(output_dir)

        # 4. Statistical comparison
        self._plot_statistics(output_dir)

        # 5. Gradient analysis (reward differences for similar IoU)
        self._plot_gradient_analysis(output_dir)

        print(f"\nAll plots saved to: {output_dir}/")

    def _plot_reward_distributions(self, output_dir: str):
        """Plot reward distributions for each function"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (rf_name, metrics) in enumerate(self.results.items()):
            rewards = [r for r in metrics.rewards if not np.isnan(r)]

            ax = axes[idx]
            ax.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(metrics.statistics['mean'], color='red',
                      linestyle='--', linewidth=2, label=f"Mean: {metrics.statistics['mean']:.3f}")
            ax.axvline(metrics.statistics['median'], color='green',
                      linestyle='--', linewidth=2, label=f"Median: {metrics.statistics['median']:.3f}")

            ax.set_xlabel('Reward Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{rf_name}\nReward Distribution', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_reward_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 01_reward_distributions.png")

    def _plot_scenario_comparison(self, output_dir: str):
        """Compare reward functions across different scenario types"""

        scenario_types = set()
        for tc in self.test_cases:
            scenario_types.add(tc.scenario_type)

        scenario_types = sorted(list(scenario_types))

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(scenario_types))
        width = 0.2

        for idx, (rf_name, metrics) in enumerate(self.results.items()):
            means = []
            for scenario in scenario_types:
                scenario_rewards = metrics.scenario_rewards.get(scenario, [])
                if scenario_rewards:
                    means.append(np.mean([r for r in scenario_rewards if not np.isnan(r)]))
                else:
                    means.append(0)

            ax.bar(x + idx * width, means, width, label=rf_name, alpha=0.8)

        ax.set_xlabel('Scenario Type', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Reward', fontsize=13, fontweight='bold')
        ax.set_title('Reward Function Performance by Scenario Type', fontsize=15, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(scenario_types, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_scenario_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 02_scenario_comparison.png")

    def _plot_heatmap(self, output_dir: str):
        """Create heatmap showing rewards for each function on each test case"""

        # Prepare data
        rf_names = list(self.results.keys())
        test_names = [tc.name for tc in self.test_cases]

        data = np.zeros((len(rf_names), len(test_names)))

        for i, rf_name in enumerate(rf_names):
            for j, reward in enumerate(self.results[rf_name].rewards):
                data[i, j] = reward if not np.isnan(reward) else 0

        # Create heatmap
        fig, ax = plt.subplots(figsize=(18, 8))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)

        ax.set_xticks(np.arange(len(test_names)))
        ax.set_yticks(np.arange(len(rf_names)))
        ax.set_xticklabels(test_names, rotation=90, ha='right', fontsize=10)
        ax.set_yticklabels(rf_names, fontsize=11)

        # Add text annotations
        for i in range(len(rf_names)):
            for j in range(len(test_names)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Reward Function Heatmap: All Test Cases',
                    fontsize=15, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Reward Value', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_reward_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 03_reward_heatmap.png")

    def _plot_statistics(self, output_dir: str):
        """Plot statistical comparison of reward functions"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Box plot
        ax1 = axes[0]
        data_for_box = []
        labels = []
        for rf_name, metrics in self.results.items():
            rewards = [r for r in metrics.rewards if not np.isnan(r)]
            data_for_box.append(rewards)
            labels.append(rf_name.replace('_', '\n'))

        bp = ax1.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax1.set_ylabel('Reward Value', fontsize=12, fontweight='bold')
        ax1.set_title('Reward Distribution (Box Plot)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # Bar plot of statistics
        ax2 = axes[1]
        rf_names = list(self.results.keys())
        x = np.arange(len(rf_names))

        means = [self.results[rf].statistics['mean'] for rf in rf_names]
        stds = [self.results[rf].statistics['std'] for rf in rf_names]

        ax2.bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue', edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels([rf.replace('_', '\n') for rf in rf_names], fontsize=10)
        ax2.set_ylabel('Mean Reward ± Std', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Reward with Standard Deviation', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_statistical_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 04_statistical_comparison.png")

    def _plot_gradient_analysis(self, output_dir: str):
        """Analyze reward gradients for different IoU levels"""

        # Create synthetic IoU gradient test
        iou_levels = np.linspace(0, 1.0, 21)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (rf_name, rf_module) in enumerate(self.reward_functions.items()):
            rewards_single = []
            rewards_multi = []

            # Test single box at different IoU levels
            for iou in iou_levels:
                # Simulate box with specific IoU
                if iou == 0:
                    pred_box = [[300, 300, 400, 400]]  # No overlap
                else:
                    # Approximate box position for target IoU
                    offset = int((1 - iou) * 50)
                    pred_box = [[100 + offset, 100 + offset, 200, 200]]

                gt_str = "[100,100,200,200]"
                pred_str = self._format_boxes_for_reward(pred_box)

                try:
                    reward = rf_module.compute_score("eval", pred_str, gt_str, {}, False)
                    rewards_single.append(reward)
                except:
                    rewards_single.append(0)

                # Test multi-box scenario
                try:
                    gt_str_multi = "[100,100,200,200] [300,300,400,400]"
                    pred_str_multi = f"{pred_str} [300,300,400,400]"
                    reward_multi = rf_module.compute_score("eval", pred_str_multi, gt_str_multi, {}, False)
                    rewards_multi.append(reward_multi)
                except:
                    rewards_multi.append(0)

            ax = axes[idx]
            ax.plot(iou_levels, rewards_single, 'o-', linewidth=2, label='Single Box', markersize=6)
            ax.plot(iou_levels, rewards_multi, 's-', linewidth=2, label='Multi Box (2)', markersize=6, alpha=0.7)
            ax.set_xlabel('IoU Level', fontsize=12)
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_title(f'{rf_name}\nReward vs IoU', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.3)

            # Add shading for IoU regions
            ax.axvspan(0, 0.3, alpha=0.1, color='red', label='Poor')
            ax.axvspan(0.3, 0.5, alpha=0.1, color='yellow')
            ax.axvspan(0.5, 0.75, alpha=0.1, color='lightgreen')
            ax.axvspan(0.75, 1.0, alpha=0.1, color='green')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_gradient_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 05_gradient_analysis.png")

    def print_summary_report(self):
        """Print comprehensive text summary"""

        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)

        for rf_name, metrics in self.results.items():
            print(f"\n{rf_name}")
            print("-" * 80)
            print(f"  Mean Reward:      {metrics.statistics['mean']:>8.4f}")
            print(f"  Std Dev:          {metrics.statistics['std']:>8.4f}")
            print(f"  Median:           {metrics.statistics['median']:>8.4f}")
            print(f"  Range:            [{metrics.statistics['min']:.4f}, {metrics.statistics['max']:.4f}]")
            print(f"  Q25-Q75:          [{metrics.statistics['q25']:.4f}, {metrics.statistics['q75']:.4f}]")

            print("\n  Scenario Breakdown:")
            for scenario_type in sorted(metrics.scenario_rewards.keys()):
                rewards = [r for r in metrics.scenario_rewards[scenario_type] if not np.isnan(r)]
                if rewards:
                    print(f"    {scenario_type:20s}: mean={np.mean(rewards):>7.4f}, "
                          f"std={np.std(rewards):>7.4f}, n={len(rewards)}")

    def save_results_json(self, output_path: str = "evaluation_results.json"):
        """Save detailed results to JSON"""

        output_data = {
            'metadata': {
                'num_reward_functions': len(self.reward_functions),
                'num_test_cases': len(self.test_cases),
                'reward_functions': list(self.reward_functions.keys())
            },
            'test_cases': [],
            'results': {}
        }

        # Save test case details
        for tc in self.test_cases:
            output_data['test_cases'].append({
                'name': tc.name,
                'description': tc.description,
                'scenario_type': tc.scenario_type,
                'num_gt_boxes': len(tc.ground_truth),
                'num_pred_boxes': len(tc.predictions)
            })

        # Save results
        for rf_name, metrics in self.results.items():
            output_data['results'][rf_name] = {
                'statistics': metrics.statistics,
                'rewards_per_test': metrics.rewards,
                'scenario_rewards': {k: v for k, v in metrics.scenario_rewards.items()}
            }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Saved detailed results to: {output_path}")

    def generate_recommendations(self):
        """Generate recommendations for GRPO training"""

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR GRPO TRAINING")
        print("=" * 80)

        # Rank by various criteria
        rankings = {
            'mean_reward': sorted(self.results.items(),
                                 key=lambda x: x[1].statistics['mean'], reverse=True),
            'stability': sorted(self.results.items(),
                               key=lambda x: x[1].statistics['std']),
            'dynamic_range': sorted(self.results.items(),
                                   key=lambda x: x[1].statistics['range'], reverse=True)
        }

        print("\n1. RANKING BY MEAN REWARD:")
        for i, (rf_name, metrics) in enumerate(rankings['mean_reward'], 1):
            print(f"   {i}. {rf_name:30s} (mean={metrics.statistics['mean']:.4f})")

        print("\n2. RANKING BY STABILITY (lowest std):")
        for i, (rf_name, metrics) in enumerate(rankings['stability'], 1):
            print(f"   {i}. {rf_name:30s} (std={metrics.statistics['std']:.4f})")

        print("\n3. RANKING BY DYNAMIC RANGE:")
        for i, (rf_name, metrics) in enumerate(rankings['dynamic_range'], 1):
            print(f"   {i}. {rf_name:30s} (range={metrics.statistics['range']:.4f})")

        print("\n" + "-" * 80)
        print("KEY CONSIDERATIONS FOR GRPO:")
        print("-" * 80)

        print("""
For GRPO (Group Relative Policy Optimization), consider:

1. **Dense Rewards**: Functions that provide smooth gradients across IoU levels
   - Avoid sparse binary rewards (good: R3, R4, R5, R6)

2. **Balanced Precision-Recall**: Critical for multi-box detection
   - Look for balanced performance across scenario types

3. **Partial Credit**: Enables learning from imperfect predictions
   - R5 explicitly provides partial credit for mediocre boxes

4. **Gradient Smoothness**: Smooth reward curves enable stable policy updates
   - R4 uses spline transformation for smoother gradients

5. **Multi-Box Scaling**: Should scale appropriately with number of boxes
   - Check multi_box scenarios in heatmap

RECOMMENDED TOP 3 FOR RADIOLOGY GROUNDING:

Based on the evaluation, consider these priorities:
   a) If gradient smoothness is critical → R4 (Smooth Gradients)
   b) If partial credit for learning → R5 (Soft Partial Credit)
   c) If balanced F1 approach → R6 (F1-Weighted IoU)
   d) If proven baseline → R3 (F-beta × mean IoU)

Examine the plots and scenario-specific performance to make final decision.
        """)


def main():
    """Main execution"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         REWARD FUNCTION EVALUATION FOR RADIOLOGY GROUNDING/DETECTION       ║
║                                                                            ║
║         Master Thesis: Optimal Reward Functions for GRPO-based VLM         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Create evaluator
    evaluator = RewardFunctionEvaluator()

    # Run evaluation
    results = evaluator.evaluate_all()

    # Generate plots
    evaluator.plot_results(output_dir="reward_evaluation_results")

    # Print summary
    evaluator.print_summary_report()

    # Save results
    evaluator.save_results_json("reward_evaluation_results/evaluation_results.json")

    # Generate recommendations
    evaluator.generate_recommendations()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nCheck the 'reward_evaluation_results/' directory for:")
    print("  - 5 comprehensive plots (PNG)")
    print("  - Detailed results (JSON)")
    print("  - This terminal output for analysis")
    print("\n")


if __name__ == "__main__":
    main()
