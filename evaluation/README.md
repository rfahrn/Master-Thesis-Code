# GPRO Reward Function Evaluation Framework

**For Master Thesis: Comparative Analysis of RL Reward Functions for Radiology Grounding**

---

## Overview

This directory contains a comprehensive evaluation framework for comparing reward functions used in GPRO (Grounded Post-training for Radiology with Reinforcement Learning). The framework enables:

1. **Systematic edge case testing** across 40+ scenarios
2. **Quantitative comparison** of reward signals
3. **Publication-quality visualizations** for thesis
4. **Mathematical documentation** of all reward functions

---

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── edge_case_test_suite.py        # Comprehensive test cases (40+ scenarios)
├── reward_function_evaluator.py   # Evaluation framework
├── visualizations.py              # Plotting and visualization
├── results/                       # Evaluation results (generated)
│   ├── evaluation_results_*.csv
│   ├── statistics_*.json
│   ├── signal_differences_*.csv
│   └── summary_report_*.txt
└── plots/                         # Generated plots (publication-ready)
    ├── reward_distributions.png
    ├── category_heatmap.png
    ├── edge_case_comparison.png
    ├── signal_correlation.png
    ├── iou_reward_curves.png
    ├── summary_statistics.png
    └── scatter_*.png
```

---

## Quick Start

### 1. Run Complete Evaluation

```bash
cd /home/user/Master-Thesis-Code

# Run evaluation on all test cases
python evaluation/reward_function_evaluator.py
```

This will:
- Evaluate all 5 reward functions (R1-R5)
- Run on 40+ edge case scenarios
- Generate CSV results, JSON statistics, and text report
- Save to `evaluation/results/`

### 2. Generate Visualizations

```bash
# After running evaluation, generate plots
python evaluation/visualizations.py \
    --results evaluation/results/evaluation_results_*.csv \
    --stats evaluation/results/statistics_*.json \
    --output evaluation/plots
```

This will generate:
- Reward distribution plots
- Category performance heatmaps
- Edge case comparison charts
- Signal correlation matrices
- IoU-reward curves
- Pairwise scatter plots

### 3. View Results

```bash
# View summary report
cat evaluation/results/summary_report_*.txt

# Open plots (if on GUI system)
xdg-open evaluation/plots/reward_distributions.png
```

---

## Test Suite

### Test Categories

The test suite includes 7 categories of edge cases:

1. **Basic Cases (4 tests)**
   - True negative (correct empty prediction)
   - Perfect match (IoU = 1.0)
   - Hallucination (false positive)
   - Missed detection (false negative)

2. **Localization Quality (5 tests)**
   - Poor localization (IoU ≈ 0.25)
   - Near-threshold (IoU ≈ 0.48)
   - Just-above-threshold (IoU ≈ 0.52)
   - Good localization (IoU ≈ 0.70)
   - Excellent localization (IoU ≈ 0.90)

3. **Multi-Box Scenarios (5 tests)**
   - Two perfect matches
   - Partial multi-box match
   - Many-to-one (multiple preds, one GT)
   - One-to-many (one pred, multiple GT)
   - Many-to-many complex

4. **Geometric Edge Cases (6 tests)**
   - Tiny box (small finding)
   - Large box (full image)
   - Vertical box (tall, narrow)
   - Horizontal box (wide, short)
   - Corner box (edge of image)
   - Zero-area box (degenerate)

5. **Overlap Scenarios (6 tests)**
   - Completely nested box
   - Fully enclosing box
   - Partial overlap - shifted right
   - Diagonal offset
   - Minimal overlap - corner touch
   - No overlap - completely separate

6. **Clinical Scenarios (3 tests)**
   - Multiple small nodules
   - Bilateral findings
   - Overlapping anatomical structures

7. **Matching Algorithm Stress Tests (3 tests)**
   - Ambiguous matches
   - Same IoU multiple matches
   - High cardinality (10+ boxes)

**Total: 40+ comprehensive test cases**

### Example Test Case

```python
{
    'id': 'iou_003',
    'category': 'localization',
    'name': 'Just-Above-Threshold (IoU ≈ 0.52)',
    'description': 'Just above matching threshold',
    'prediction': '<answer>[0.105, 0.205, 0.305, 0.405]</answer>',
    'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
    'expected_behavior': 'Above threshold, counts as match',
    'clinical_scenario': 'Acceptable lesion localization',
    'edge_case_type': 'one_to_one',
    'expected_iou': 0.52
}
```

---

## Reward Functions Evaluated

### R1: Average Precision at IoU 0.5
- **Type:** AP@0.5
- **Matching:** Sorted greedy
- **Hyperparameters:** 1 (NO_BOX_BONUS)
- **Best for:** Simple baseline, standard metric

### R2: F-beta × IoU Baseline
- **Type:** F-beta × product of IoUs
- **Matching:** Greedy
- **Hyperparameters:** 3 (BETA, MIN_IOU, NO_BOX_BONUS)
- **Best for:** Single-box tasks (poor multi-box performance)

### R3: F-beta × mean_IoU
- **Type:** F-beta × arithmetic mean IoU
- **Matching:** Sorted greedy
- **Hyperparameters:** 3 (BETA, MIN_IOU, NO_BOX_BONUS)
- **Best for:** Production deployment, multi-box tasks

### R4: Enhanced Smooth Spline
- **Type:** F-beta × smooth spline-transformed IoU
- **Matching:** Sorted greedy
- **Hyperparameters:** 4-5 (BETA, MIN_IOU, NO_BOX_BONUS, CENTER_WEIGHT)
- **Best for:** RL training (best gradients)

### R5: Strict Medical Grounding
- **Type:** F-beta × piecewise quality function
- **Matching:** Hungarian (optimal)
- **Hyperparameters:** 5 (BETA, MIN_IOU, NO_BOX_BONUS, quality breakpoints)
- **Best for:** Medical applications (strictest, optimal matching)

---

## Evaluation Metrics

### Per-Function Metrics

- **Mean reward:** Average reward across all test cases
- **Std reward:** Standard deviation (stability measure)
- **Median reward:** Robust central tendency
- **Min/Max reward:** Range of rewards
- **Percentiles:** 25th, 50th, 75th, 90th, 95th
- **Success rate:** % of evaluations without errors

### Cross-Function Metrics

- **Signal correlation:** Pearson correlation between reward signals
- **Mean difference:** Average reward difference
- **Abs mean difference:** Average absolute difference
- **Max/min difference:** Range of differences

### Category-Wise Metrics

- **Mean by category:** Performance on each test category
- **Std by category:** Variability within category
- **Count by category:** Number of tests per category

### Edge Case Type Metrics

- **Mean by edge case:** Performance on each scenario type
- **Std by edge case:** Variability for each scenario
- **Count by edge case:** Number of tests per scenario

---

## Output Files

### 1. Evaluation Results CSV

**File:** `evaluation_results_YYYYMMDD_HHMMSS.csv`

Columns:
- `reward_function`: R1, R2, R3, R4, R5
- `rf_full_name`: Full descriptive name
- `rf_type`: Mathematical type (e.g., "F-beta × mean_IoU")
- `rf_matching`: Matching algorithm used
- `rf_complexity`: Complexity level
- `test_id`: Unique test identifier
- `test_name`: Human-readable test name
- `category`: Test category
- `edge_case_type`: Scenario type
- `clinical_scenario`: Clinical interpretation
- `reward`: Computed reward value
- `success`: Whether evaluation succeeded
- `error`: Error message (if failed)
- `prediction`: Model prediction string
- `ground_truth`: Ground truth string
- `details`: JSON string with detailed metrics

### 2. Statistics JSON

**File:** `statistics_YYYYMMDD_HHMMSS.json`

Structure:
```json
{
  "R1": {
    "overall": {
      "mean": 0.487,
      "std": 0.369,
      "min": 0.0,
      "max": 1.0,
      "median": 0.500,
      "q25": 0.200,
      "q75": 0.800,
      "success_rate": 1.0
    },
    "by_category": {
      "basic": {"mean": 0.550, "std": 0.432, "count": 4},
      ...
    },
    "by_edge_case_type": {
      "one_to_one": {"mean": 0.623, "std": 0.301, "count": 15},
      ...
    }
  },
  "R2": { ... },
  ...
}
```

### 3. Signal Differences CSV

**File:** `signal_differences_YYYYMMDD_HHMMSS.csv`

Columns:
- `comparison`: E.g., "R1 - R3"
- `rf1`, `rf2`: Functions being compared
- `mean_diff`: Mean of (rf1 - rf2)
- `std_diff`: Std of differences
- `abs_mean_diff`: Mean of |rf1 - rf2|
- `max_diff`, `min_diff`: Range of differences
- `correlation`: Pearson correlation

### 4. Summary Report TXT

**File:** `summary_report_YYYYMMDD_HHMMSS.txt`

Human-readable report with:
- Overall statistics per function
- Category-wise performance comparison
- Pairwise signal differences
- Key findings and insights

---

## Visualization Plots

### 1. Reward Distributions (`reward_distributions.png`)

Two subplots:
- **Top:** Violin plots showing overall reward distribution per function
- **Bottom:** Box plots showing distribution by category and function

### 2. Category Heatmap (`category_heatmap.png`)

Heatmap showing mean reward for each (category, function) pair.
- Rows: Test categories
- Columns: Reward functions
- Color: Mean reward (green = high, red = low)

### 3. Edge Case Comparison (`edge_case_comparison.png`)

Grouped bar chart comparing mean rewards by edge case type.

### 4. Signal Correlation (`signal_correlation.png`)

Correlation matrix heatmap showing how similar reward signals are between functions.

### 5. IoU-Reward Curves (`iou_reward_curves.png`)

Four subplots:
- **Top-left:** All functions overlaid
- **Top-right:** R1 threshold behavior
- **Bottom-left:** R4 smooth spline behavior
- **Bottom-right:** Reward variance comparison

### 6. Summary Statistics (`summary_statistics.png`)

Four subplots:
- **Top-left:** Mean rewards with error bars
- **Top-right:** Median rewards
- **Bottom-left:** Reward ranges (min to max)
- **Bottom-right:** Coefficient of variation

### 7. Pairwise Scatter Plots (`scatter_RX_vs_RY.png`)

Scatter plots comparing two functions:
- X-axis: Reward from function X
- Y-axis: Reward from function Y
- Colors: Test categories
- Diagonal line: y=x reference

Generated for key comparisons:
- R1 vs R3 (Baseline vs Production)
- R3 vs R4 (Production vs Enhanced)
- R4 vs R5 (Enhanced vs Strict)
- R1 vs R5 (Baseline vs Strict)

---

## Usage Examples

### Run Subset of Tests

```python
from evaluation.edge_case_test_suite import EdgeCaseTestSuite
from evaluation.reward_function_evaluator import RewardFunctionEvaluator

# Load test suite
suite = EdgeCaseTestSuite()

# Get specific category
localization_tests = suite.get_tests_by_category('localization')

# Or get specific test
test = suite.get_test_by_id('iou_003')
```

### Evaluate Single Test Case

```python
from Reward_Functions import R1, R3, R4

# Get a test case
suite = EdgeCaseTestSuite()
test = suite.get_test_by_id('basic_002')  # Perfect match

# Evaluate with different functions
r1_result = R1.compute_score(
    data_source="test",
    solution_str=test['prediction'],
    ground_truth=test['ground_truth'],
    return_details=True
)

r3_result = R3.compute_score(
    data_source="test",
    solution_str=test['prediction'],
    ground_truth=test['ground_truth'],
    return_details=True
)

print(f"R1 reward: {r1_result['reward']}")
print(f"R3 reward: {r3_result['reward']}")
```

### Generate Custom Plot

```python
from evaluation.visualizations import RewardFunctionVisualizer

# Load results
viz = RewardFunctionVisualizer(
    results_csv="evaluation/results/evaluation_results_*.csv",
    stats_json="evaluation/results/statistics_*.json"
)

# Generate specific plot
viz.plot_category_heatmap("custom_heatmap.png")

# Or generate all plots
viz.generate_all_plots("evaluation/plots")
```

---

## Interpreting Results

### What Makes a Good Reward Function?

1. **High mean reward on correct predictions** (perfect matches, good IoU)
2. **Low reward on incorrect predictions** (hallucinations, missed detections)
3. **Smooth gradient w.r.t. IoU** (for RL training)
4. **Stable across categories** (low variance)
5. **Discriminative:** Clear separation between good and bad predictions
6. **Multi-box friendly:** Doesn't penalize multiple correct detections

### Red Flags

- **Very low correlation with other functions:** May be measuring something different
- **High variance within categories:** Inconsistent behavior
- **Poor performance on "one_to_one" cases:** Basic functionality issue
- **Rewards hallucinations highly:** False positive problem

### Thesis Recommendations

Based on evaluation results, your thesis should include:

1. **Table:** Mean ± Std for each function across all tests
2. **Figure:** Category heatmap showing strengths/weaknesses
3. **Figure:** Correlation matrix showing signal similarity
4. **Figure:** IoU-reward curves showing gradient properties
5. **Table:** Pairwise differences and correlations
6. **Discussion:** Which function for which use case

---

## Extending the Framework

### Adding New Test Cases

Edit `edge_case_test_suite.py`:

```python
{
    'id': 'custom_001',
    'category': 'custom',
    'name': 'Your Test Name',
    'description': 'What this tests',
    'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
    'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
    'expected_behavior': 'What should happen',
    'clinical_scenario': 'Clinical context',
    'edge_case_type': 'one_to_one'
}
```

### Adding New Reward Function

1. Implement in `Reward_Functions/R6.py`
2. Add to evaluator in `reward_function_evaluator.py`:

```python
self.reward_functions = {
    ...
    'R6': {
        'module': R6,
        'name': 'Your Function Name',
        'description': 'Description',
        'type': 'Mathematical Type',
        'matching': 'Algorithm Used',
        'complexity': 'Number of params'
    }
}
```

### Adding New Visualizations

Edit `visualizations.py` and add method:

```python
def plot_custom_analysis(self, output_path: str):
    """Your custom plot."""
    # Implementation
    plt.savefig(output_path)
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure parent directory is in Python path
export PYTHONPATH="/home/user/Master-Thesis-Code:$PYTHONPATH"
```

### Missing Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Evaluation Fails

Check:
1. All reward function files exist in `Reward_Functions/`
2. Each has `compute_score()` function
3. Test cases have correct format

### Visualization Fails

Ensure evaluation was run first:
```bash
ls evaluation/results/evaluation_results_*.csv
```

---

## Performance Notes

- **Evaluation time:** ~2-5 minutes for all 5 functions on 40 tests
- **Memory usage:** <1 GB
- **Plot generation:** ~30 seconds for all plots

---

## Citation

If you use this framework in your thesis or publications, please cite:

```bibtex
@mastersthesis{yourname2025gpro,
  title={Comparative Analysis of RL Reward Functions for Radiology Grounding},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

---

## Contact

For questions or issues:
- Check the main thesis repository README
- Review mathematical definitions in `docs/MATHEMATICAL_DEFINITIONS.md`
- Review matching algorithms in `docs/MATCHING_ALGORITHMS.md`

---

**Last Updated:** 2025-10-29
**Version:** 1.0
