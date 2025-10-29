# Comprehensive Reward Functions Summary for GPRO Training

## Overview
This document provides a detailed analysis of all reward functions in the GPRO (Group Relative Policy Optimization) training framework for radiology grounding tasks. The reward functions are organized by complexity and designed to handle multi-bounding box detection in vision-language models.

---

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Reward Functions Overview](#reward-functions-overview)
3. [Common Utilities & Helper Functions](#common-utilities--helper-functions)
4. [Detailed Function Analysis](#detailed-function-analysis)
5. [Mathematical Approaches](#mathematical-approaches)
6. [Bounding Box Matching Strategies](#bounding-box-matching-strategies)
7. [Evaluation & Testing Framework](#evaluation--testing-framework)
8. [Recommendations](#recommendations)

---

## Directory Structure

```
/home/user/Master-Thesis-Code/
├── Reward_Functions/
│   ├── R1.py                 (Simple AP@0.5 Baseline)
│   ├── R2.py                 (F-beta × IoU)
│   ├── R3.py                 (RL-Optimized F-beta × mean_IoU)
│   ├── R4.py                 (Enhanced Smooth Gradients)
│   ├── R5.py                 (Soft Partial Credit / R7 Simplified)
│   └── README.md             (Comprehensive guide)
├── evaluate_reward_functions.py  (Evaluation framework)
├── REWARD_FUNCTION_ANALYSIS.md   (Detailed analysis & recommendations)
├── test_r1_simple.py             (Simple test without dependencies)
└── grpo_simulation.py             (Integration framework)
```

---

## Reward Functions Overview

| Function | Complexity | Approach | Best For | Hyperparameters |
|----------|-----------|----------|----------|-----------------|
| **R1** | Simplest | Binary threshold at IoU=0.5 | Baseline/evaluation metric matching | 1 (NO_BOX_BONUS) |
| **R2** | Simple | F-β score × mean_IoU with greedy matching | Clean GRPO baseline | 3 (BETA, MIN_IOU, NO_BOX_BONUS) |
| **R3** | Simple | Dense F-β × mean_IoU reward | General GRPO training | 3 (BETA=1.0, MIN_IOU=0.5, NO_BOX_BONUS=0.2) |
| **R4** | Medium | F-β × smooth quality (cubic spline) | Advanced GRPO with better gradients | 4-5 (includes USE_CENTER_AWARE) |
| **R5** | Complex | F-β with piecewise linear partial credit | Smooth training with exploration | 5 (β, IOU_THRESHOLD, penalties, etc.) |

---

## Common Utilities & Helper Functions

All reward functions share common utility functions with identical implementations:

### 1. **extract_bounding_boxes(answer: str) -> List[List[float]]**
```python
# Regex pattern to extract [x1, y1, x2, y2] from answer text
NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"

# Features:
- Supports scientific notation (e.g., 1.5e-2)
- Handles negative coordinates
- Validates coordinates are finite (not NaN/Inf)
- Returns empty list if no boxes found
```

### 2. **compute_iou(box1: List[float], box2: List[float]) -> float**
```python
# Standard IoU calculation
# box format: [x1, y1, x2, y2]

def compute_iou(box1, box2) -> float:
    # Find intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    
    # Compute union area
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = box1_area + box2_area - inter_area
    
    return inter_area / union if union > 0 else 0.0
```

### 3. **classify_edge_case(n_pred: int, n_gt: int) -> str**
Used in R1, R3, R4 for analysis:
- `true_negative`: No predictions, no GT boxes
- `hallucination`: Predictions made but no GT
- `missed_detection`: No predictions but GT exists
- `one_to_one`: Single prediction, single GT
- `many_to_one`: Multiple predictions, one GT
- `one_to_many`: One prediction, multiple GT
- `many_to_many`: Multiple predictions and GT

### 4. **greedy_match_boxes(pred_boxes, actual_boxes, min_iou)**
Fast greedy matching algorithm (O(n² log n)):
1. Compute IoU matrix between all predictions and GT boxes
2. Sort predictions by maximum IoU (descending)
3. For each prediction, match to best available (unmatched) GT box
4. Only accept match if IoU >= min_iou threshold
5. Return list of (pred_idx, gt_idx, iou) tuples

**Why Greedy Over Hungarian?**
- 15x faster for n≈100 boxes
- Usually produces same optimal results
- More stable for RL training

---

## Detailed Function Analysis

### R1: Simple AP@0.5 Baseline

**File:** `/home/user/Master-Thesis-Code/Reward_Functions/R1.py`

**Purpose:** Matches standard AP@0.5 evaluation metric used in COCO/radiology benchmarks

**Key Features:**
- Binary threshold at IoU=0.5 (match or no match)
- Sorted greedy matching (processes best IoU matches first)
- Edge case classification and analysis
- Detailed metrics reporting

**Mathematical Formula:**
```
reward = AP@0.5 (binary: IoU ≥ 0.5 → match, else no match)

REWARD DISTRIBUTION:
├─ Both empty (true negative)     → NO_BOX_BONUS (0.2)
├─ Pred empty, GT exists (FN)     → 0.0
├─ Pred exists, GT empty (FP)     → 0.0
└─ Both exist                       → AP@0.5 computed
```

**Hyperparameters:**
- `NO_BOX_BONUS = 0.2` (reward for correct negatives)

**Matching Algorithm:**
```python
def average_precision_at_iou(pred_boxes, actual_boxes, iou_threshold=0.5):
    # 1. Compute IoU matrix: ious[i,j] = IoU(pred_i, actual_j)
    # 2. Sort predictions by max IoU (descending)
    # 3. Greedy matching with IoU threshold
    # 4. Compute precision and recall
    # 5. Return AP from precision-recall curve
```

**Edge Case Handling:**
```python
├─ no boxes predicted, no GT         → reward = 0.2 (correct negative)
├─ boxes predicted, but no GT        → reward = 0.0 (hallucination)
├─ no boxes predicted, GT has boxes  → reward = 0.0 (missed)
├─ boxes predicted, GT has boxes     → AP@0.5 (0.0 to 1.0)
│   ├─ perfect localization          → 1.0
│   ├─ good localization (IoU≥0.5)   → 0.5 to 1.0
│   ├─ poor localization (IoU<0.5)   → 0.0 to 0.5
│   └─ completely wrong              → 0.0
```

**Key Parameters:**
- `iou_threshold`: 0.5 (COCO standard)
- `NO_BOX_BONUS`: 0.2 (intentionally low to encourage localization)

**Strengths:**
✅ Directly matches evaluation metric (AP@0.5)
✅ Simple and deterministic
✅ Well-established in computer vision

**Weaknesses:**
❌ Hard threshold at 0.5 (IoU=0.49 → 0, IoU=0.50 → counts)
❌ No dense rewards for partial credit
❌ Binary matching may miss learning signal

---

### R2: F-beta × IoU (RL-Optimized)

**File:** `/home/user/Master-Thesis-Code/Reward_Functions/R2.py`

**Purpose:** Clean GRPO baseline combining detection and localization quality

**Key Features:**
- Separates detection quality (F-β) from localization quality (mean_IoU)
- Dense rewards (no hard cutoffs)
- Natural penalties through precision/recall balance
- Well-balanced for RL training

**Mathematical Formula:**
```
reward = F_β × mean_IoU

Where:
- F_β = (1+β²) × (P×R) / (β²×P + R)
  ├─ Precision = num_matches / num_predictions
  ├─ Recall = num_matches / num_ground_truths
  └─ β ∈ [1.0, 2.0]: 1.0=balanced, 1.5=mild recall, 2.0=strong recall
  
- mean_IoU = average IoU of matched boxes [0, 1]
```

**Hyperparameters:**
- `BETA = 1.0` (F-beta parameter: balanced F1 score)
- `MIN_IOU_THRESHOLD = 0.5` (COCO standard matching threshold)
- `NO_BOX_BONUS = 0.2` (true negative reward)

**Matching Algorithm:**
Greedy matching with minimum IoU threshold

**Edge Cases:**
```python
├─ Both empty (pred=0, GT=0)        → NO_BOX_BONUS (0.2)
├─ Only pred empty (pred=0, GT>0)   → F_β=0 → reward=0
├─ Only GT empty (pred>0, GT=0)     → F_β=0 → reward=0
└─ Both non-empty                    → F_β × mean_IoU
```

**Advantages for GRPO:**
✅ Dense rewards: Every IoU value contributes
✅ Clear credit assignment: F-β for detection, mean_IoU for localization
✅ Natural penalties: F-beta precision term handles over-prediction
✅ Bounded [0,1]: Stable for RL training
✅ Fast: Greedy matching

**Typical Rewards:**
```
Perfect match (IoU=1.0, all matched)     → F_β=1.0 × 1.0 = 1.0
Good detection (IoU=0.7, all matched)    → F_β=1.0 × 0.7 = 0.7
Partial (50% matched, IoU=0.6 avg)       → F_β=0.67 × 0.6 = 0.40
Hallucination (extra false positive)     → F_β reduced by precision penalty
```

---

### R3: F-beta × mean_IoU (Cleaner Version of R2)

**File:** `/home/user/Master-Thesis-Code/Reward_Functions/R3.py`

**Purpose:** Clean, production-ready GRPO baseline

**Key Quote from Code:**
> "Why this design for RL/GRPO: Dense rewards, clear credit assignment, stable gradients, natural penalties, fast matching"

**Key Features:**
- Nearly identical to R2 but with cleaner documentation
- Emphasizes why this design is optimal for RL
- Comprehensive test cases and analysis tools

**Mathematical Formula:**
```
reward = F_β × mean_IoU

Same as R2, but with extensive documentation on why this is RL-optimal
```

**Hyperparameters:**
- `BETA = 1.0` (balanced)
- `MIN_IOU_THRESHOLD = 0.5` (COCO standard)
- `NO_BOX_BONUS = 0.2`

**Key Innovation:** Extensive analysis of why this simple approach is better than complex multi-parameter designs

**Comparison with Complex Approaches:**
```
COMPLEX REWARD:
- 10+ hyperparameters → hard to tune
- Multiple penalties (sparse signal)
- Redundant credit assignment
- Slow Hungarian matching O(n³)

R3 SIMPLE APPROACH:
- 3 hyperparameters → easy to tune
- Single formula (dense signal)
- Clear credit assignment
- Fast greedy O(n² log n)
```

**Why R3 Over R2:**
Functionally identical, but R3 has clearer documentation and stronger emphasis on RL optimization principles.

---

### R4: Enhanced Smooth Gradients

**File:** `/home/user/Master-Thesis-Code/Reward_Functions/R4.py`

**Purpose:** Advanced GRPO training with better gradient shaping

**Key Innovation:** Cubic Hermite spline transformation of IoU values for smoother reward curves

**Mathematical Formula:**
```
reward = F_β × smooth_quality

Where:
- F_β = standard F-beta score
- smooth_quality = mean(spline(IoU_i))
  └─ Uses cubic Hermite interpolation with control points:
     ├─ (0.0, 0.0, grad=1.5)  - steep initial gradient
     ├─ (0.5, 0.5, grad=1.1)  - at threshold
     ├─ (0.8, 0.8, grad=0.9)  - good zone
     └─ (1.0, 1.0, grad=0.5)  - diminishing returns

OPTIONAL: Center-aware quality = (1-w) × smooth_IoU + w × center_quality
```

**Hyperparameters:**
- `BETA = 1.5` (mild recall emphasis for medical)
- `MIN_IOU_THRESHOLD = 0.5`
- `NO_BOX_BONUS = 0.2`
- `USE_CENTER_AWARE = False` (optional enhancement)
- `CENTER_WEIGHT = 0.15` (if center-aware enabled)

**Smooth Reward Function (Class SmoothRewardSpline):**

The spline transformation creates different gradient magnitudes:
```
IoU ranges    Linear (R3)    Spline (R4)    Gradient Ratio
0.0 → 0.1     0.0 → 0.1      0.0 → 0.18      1.8x steeper
0.3 → 0.4     0.3 → 0.4      0.38 → 0.48     1.0x steeper
0.5 → 0.6     0.5 → 0.6      0.56 → 0.63     0.7x gentler
0.9 → 1.0     0.9 → 1.0      0.93 → 1.0      0.7x gentler
```

**Benefits Over R3:**
```
1. SMOOTHER GRADIENTS
   - Linear mean_IoU has constant gradient (1.0)
   - Spline has varying gradients (better learning dynamics)
   - C¹ continuous (no gradient jumps)

2. REWARD SHAPING
   - Steeper early [0-0.5]: Encourages crossing threshold
   - Gentler late [0.8-1.0]: Avoids perfectionism
   - Consistent signal magnitude

3. OPTIONAL GEOMETRIC REFINEMENT
   - Center distance penalty distinguishes center-aligned vs offset boxes
   - Both might have same IoU but different quality
   - Only enable if empirically needed

4. STILL RL-OPTIMIZED
   ✅ Bounded [0,1]
   ✅ Smooth (C¹ continuous)
   ✅ No discontinuities
   ✅ Clear credit assignment
   ✅ Fast (greedy matching)
```

**Center-Aware Quality (Optional):**
```python
def compute_center_distance_penalty(pred_box, gt_box) -> float:
    # Normalized by GT box diagonal for scale-invariance
    # Returns penalty [0, 1] where 0=perfect alignment, 1=very offset
    
    pred_center = ((pred[0]+pred[2])/2, (pred[1]+pred[3])/2)
    gt_center = ((gt[0]+gt[2])/2, (gt[1]+gt[3])/2)
    
    euclidean_dist = sqrt((pred_cx-gt_cx)² + (pred_cy-gt_cy)²)
    diagonal = sqrt(gt_w² + gt_h²)
    
    penalty = euclidean_dist / diagonal
    return clamp(penalty, 0, 1)
```

**Matching Algorithm:**
Same greedy matching as R2/R3

**Edge Cases:**
```
├─ Both empty (pred=0, GT=0)        → NO_BOX_BONUS (0.2)
├─ Either empty                      → reward=0
└─ Both non-empty                    → F_β × smooth_quality
```

**Strengths:**
✅ Smooth gradients improve learning stability
✅ Steeper gradient near threshold encourages crossing
✅ Optional center awareness for geometric refinement
✅ Only 4-5 hyperparameters (still simple)
✅ Still fast (greedy matching)
✅ Requires scipy for spline (standard library)

**Weaknesses:**
❌ Slightly more complex than R2/R3
❌ Requires scipy dependency
❌ May overfit on gradient shapes

**Best For:**
- Advanced GRPO training with better learning dynamics
- Models that benefit from smooth reward curves
- Medical imaging where precision matters

---

### R5: Strict Medical Focus (Hungarian Matching)

**File:** `/home/user/Master-Thesis-Code/Reward_Functions/R5.py`

**Purpose:** Most complex reward function for strict medical applications

**Key Innovation:** Hungarian algorithm for globally optimal matching

**Mathematical Formula:**
```
reward = F_β × matched_quality + penalties + bonuses

Where:
- IoU_reward(IoU) = piecewise linear with 3 regions:
  ├─ IoU < 0.15:                0.0
  ├─ 0.15 ≤ IoU < τ:           0.5 × (IoU-0.15)/(τ-0.15)
  └─ IoU ≥ τ:                   0.5 + 0.5 × (IoU-τ)/(1-τ)
  
- penalty(k) = exp(-α × k)  where k = number of errors
  
- F_β with weighted TP + penalties + recall_bonus
```

**Hyperparameters (Simplified to 5):**
- `BETA = 1.0` (can use β < 1 for precision focus)
- `IOU_THRESHOLD = 0.5` (minimum IoU for good detection)
- `PENALTY_STRENGTH = 0.7` (error harshness, 0=gentle, 1=harsh)
- `NO_FINDINGS_REWARD = 0.3` (reward for correct no-findings)
- `RECALL_BONUS = 0.1` (bonus for complete detection)

**Internal Parameters (Fixed):**
- `MIN_BOX_SIZE = 0.001` (minimum normalized box size)
- `MIN_IOU_FOR_REWARD = 0.15` (below this → zero reward)

**Matching Algorithm:**
Hungarian Algorithm (scipy.optimize.linear_sum_assignment)
```python
def hungarian_matching(pred_boxes, gt_boxes):
    # 1. Build IoU matrix (M × N)
    # 2. Convert to cost matrix: cost = 1.0 - IoU
    # 3. Apply Hungarian algorithm
    # 4. Handle unequal sizes by padding
    # 5. Return optimal matches and IoU values
```

**Why Hungarian Over Greedy?**
```
GREEDY (R1-R4):
- Fast O(n² log n)
- Usually optimal
- Simpler for RL

HUNGARIAN (R5):
- Globally optimal assignment
- Slower O(n³)
- Better for strict medical applications
```

**Edge Cases:**
```
case_1: No GT, no pred           → reward = NO_FINDINGS_REWARD
case_2: No GT, but pred exist    → penalty(M) × 0.5 (harsh for FP)
case_3: GT exists, no pred       → reward = 0 (missed findings)
case_4: Normal detection         → F_β + penalties + bonus
  ├─ Overprediction (M > N×1.5)  → F_β × penalty(excess)
  ├─ Recall bonus (all found)    → F_β + RECALL_BONUS
  └─ Standard                    → F_β
```

**Piecewise Linear IoU Reward:**
```python
def simple_iou_reward(iou, threshold=0.5) -> float:
    if iou < 0.15:
        return 0.0
    elif iou < threshold:
        # Linear ramp to 0.5 at threshold
        return 0.5 * (iou - 0.15) / (threshold - 0.15)
    else:
        # Linear from 0.5 to 1.0
        return 0.5 + 0.5 * (iou - threshold) / (1 - threshold)
```

**Penalties:**
```python
def compute_penalty(num_errors, strength=0.7) -> float:
    # Exponential decay: exp(-strength × num_errors)
    # With strength=0.7:
    # - 1 error: 0.50
    # - 2 errors: 0.25
    # - 3 errors: 0.12
    return exp(-strength * num_errors)
```

**Strengths:**
✅ Globally optimal matching (Hungarian)
✅ Strict medical focus
✅ Unified penalty system
✅ Recall bonus for complete detection
✅ Can use β=0.5 for precision emphasis
✅ Only 5 core hyperparameters

**Weaknesses:**
❌ Slowest (O(n³) Hungarian algorithm)
❌ More complex penalty system
❌ May be too strict for exploration phase

**Best For:**
- Medical applications requiring minimal false positives
- Production systems with strict quality requirements
- Final optimization phase

---

## Mathematical Approaches

### 1. Average Precision (R1)
```
AP = sum((Recall_i+1 - Recall_i) × Precision_i+1)

Where:
- Precision_i = TP_i / (TP_i + FP_i)
- Recall_i = TP_i / (TP_i + FN_i)
- Computed with binary IoU threshold (0.5)
```

### 2. F-beta Score (R2, R3, R4, R5)
```
F_β = (1 + β²) × (P × R) / (β² × P + R)

Where:
- P = Precision = TP / (TP + FP)
- R = Recall = TP / (TP + FN)
- β ∈ [0.5, 2.0]:
  - β=0.5: Emphasis precision (2x weight)
  - β=1.0: Balanced (F1 score)
  - β=1.5: Mild recall emphasis
  - β=2.0: Strong recall emphasis (4x weight)
```

### 3. Cubic Hermite Spline (R4)
```
smooth_quality = mean(spline(IoU_i))

Where spline uses control points:
- x values: [0.0, 0.5, 0.8, 1.0]
- y values: [0.0, 0.5, 0.8, 1.0]
- gradients: [1.5, 1.1, 0.9, 0.5]

Creates C¹ continuous (smooth) function with varied gradients
```

### 4. Piecewise Linear Quality (R5)
```
quality(IoU) = {
    0.0                           if IoU < 0.15
    0.5 × (IoU-0.15)/(τ-0.15)    if 0.15 ≤ IoU < τ  (ramp to 0.5)
    0.5 + 0.5 × (IoU-τ)/(1-τ)    if IoU ≥ τ          (ramp to 1.0)
}
```

---

## Bounding Box Matching Strategies

### Strategy 1: Sorted Greedy Matching (R1, R2, R3, R4)

**Algorithm:**
```
1. Compute IoU matrix: ious[i,j] = IoU(pred_i, actual_j)
2. For each prediction i:
   - Find max_iou_i = max(ious[i,:])
3. Sort predictions by max_iou (descending)
4. For each prediction in sorted order:
   - Find best unmatched GT with IoU >= min_iou
   - If found, mark both as matched
5. Return matches and IoU values
```

**Complexity:** O(n² log n)
**Optimality:** Usually optimal for most cases
**Why Use:** Fast, simple, good for RL

### Strategy 2: Hungarian Algorithm (R5)

**Algorithm:**
```
1. Build IoU matrix: ious[i,j] = IoU(pred_i, actual_j)
2. Convert to cost matrix: cost = 1.0 - ious
3. Pad matrices to square if unequal dimensions
4. Run linear_sum_assignment(cost_matrix)
5. Return optimal matches
```

**Complexity:** O(n³)
**Optimality:** Guaranteed globally optimal
**Why Use:** Medical applications, strict requirements

### Comparison

| Aspect | Greedy | Hungarian |
|--------|--------|-----------|
| Complexity | O(n² log n) | O(n³) |
| Optimality | Usually optimal | Guaranteed optimal |
| Speed | 15x faster | Slower |
| Implementation | Simple | More complex |
| RL Suitability | Better | Slower convergence |
| Medical | Acceptable | Better |

---

## Evaluation & Testing Framework

### 1. **evaluate_reward_functions.py**

Comprehensive evaluation comparing R3, R4, R5, R6 across 15 test scenarios.

**Test Cases Include:**
```
Perfect Matches (2):
├─ Perfect single box
└─ Perfect 3-box match

Partial Overlaps (4):
├─ High overlap (IoU≈0.8)
├─ Moderate overlap (IoU≈0.6)
├─ Low overlap (IoU≈0.3)
└─ Barely overlap (IoU≈0.15)

Error Cases (3):
├─ False positive (hallucination)
├─ False negative (missed detection)
└─ True negative (correct no boxes)

Multi-Box Scenarios (4):
├─ Partial recall (2/3 detected)
├─ Imprecise localization (varying IoU)
├─ Extra false positive (2 correct + 1 hallucinated)
└─ Complex (5 GT, 4 pred with varying IoU)

Stress Tests (2):
├─ Many false positives (1 GT, 5 pred)
└─ Many false negatives (5 GT, 1 pred)
```

**Key Results from Analysis:**

```
Performance Rankings:
1. R5: 0.4935 (highest mean)
2. R4: 0.4896 (good mean)
3. R3: 0.4866 (stable)
4. R6: 0.4866 (stable)

Stability Rankings (lower=better):
1. R3: 0.3691 (most stable) ⭐
2. R6: 0.3691 (most stable) ⭐
3. R4: 0.3707
4. R5: 0.3714

Complex Multi-Box Performance:
- R5: 0.6688 ⭐ (significantly better)
- R3: 0.5650
- R4: 0.5456
- R6: 0.5650

False Negative Handling (safety):
- R4: 0.2653 ⭐ (most strict)
- R3: 0.3333
- R5: 0.3333
- R6: 0.3333
```

### 2. **test_r1_simple.py**

Lightweight test without numpy/matplotlib dependencies:
```
Test Scenarios:
├─ Perfect Match
├─ True Negative
├─ Hallucination
├─ Missed Detection
├─ Partial Overlap (Good)
├─ Poor Overlap
├─ Multiple Boxes (All Correct)
└─ One-to-Many
```

**Key Function:**
```python
def compute_map_simple(pred_boxes, gt_boxes, iou_threshold=0.5):
    # Simplified mAP calculation without numpy
    # Used for validation
```

### 3. **REWARD_FUNCTION_ANALYSIS.md**

Detailed analysis with recommendations by training phase:

```
Phase 1: Early Training (Exploration)
→ R5 (Soft Partial Credit)
  ├─ Partial credit for learning
  ├─ Lower threshold (0.2)
  └─ Highest mean reward

Phase 2: Mid Training (Refinement)
→ R4 (Smooth Gradients)
  ├─ Smooth learning dynamics
  ├─ Better gradient shaping
  └─ Good safety profile

Phase 3: Final Training (Optimization)
→ R3 or R6 (F-beta or F1 Weighted)
  ├─ Most stable (std=0.3691)
  ├─ Well-proven
  └─ Better reproducibility
```

---

## Recommendations

### Single Best Choice: R4 - Enhanced Smooth Gradients

**Justification:**
1. Best overall balance (mean=0.4896, 2nd highest)
2. Superior gradient smoothness for GRPO
3. Strict on false negatives (clinical safety)
4. Good multi-box performance
5. Only 4-5 hyperparameters (manageable)

**Configuration:**
```python
from Reward_Functions import R4

reward = R4.compute_score(
    data_source="training",
    solution_str=model_output,  # e.g., "[x1,y1,x2,y2]"
    ground_truth=ground_truth_str,
    return_details=False
)
```

### Alternative Strategies

**For Maximum Learning Speed:**
→ R5 (Soft Partial Credit)
- Partial credit mechanism
- Lower MIN_IOU threshold (0.2)
- Highest mean reward (0.4935)

**For Maximum Stability:**
→ R3 or R6 (F-beta or F1 Weighted)
- Lowest variance (0.3691)
- Most reproducible
- Well-proven baselines

**Curriculum Learning (3-Phase):**
```
Phase 1 (0-20% training):  R5 with MIN_IOU=0.2
    ↓
Phase 2 (20-70% training): R4 with spline transform
    ↓
Phase 3 (70-100%):         R3 for final optimization
```

### Quick Selection Guide

```
Choose R1 if:
- Need to match COCO evaluation metric exactly
- Want simplest possible implementation
- Doing baseline comparison

Choose R2/R3 if:
- Starting general GRPO training
- Want clean, simple baseline
- F-beta × mean_IoU is sufficient

Choose R4 if:
- Want better gradient shaping
- Need smooth learning curves
- Can afford 4-5 hyperparameters
- Working on medical imaging (R4 strict on FN)

Choose R5 if:
- Need partial credit for learning
- Want fastest initial convergence
- Can tune complex penalties
- Early exploration phase
```

---

## Integration with GRPO

All reward functions implement the standard interface:

```python
def compute_score(
    data_source: str,           # Dataset name
    solution_str: str,          # Model output with <answer>...</answer>
    ground_truth: str,          # GT boxes "[x1,y1,x2,y2],[...]"
    extra_info=None,            # Optional metadata
    return_details: bool=False  # For detailed analysis
) -> float | Dict[str, Any]:
    """
    Returns:
    - float: Reward score [0, 1] (or 0.2 for true negative)
    - Dict: Detailed metrics if return_details=True
    """
```

### Expected Model Output Format

```
<answer>[x1, y1, x2, y2]</answer>      # Single box
<answer>[x1, y1, x2, y2], [...]</answer>  # Multiple boxes
<answer></answer>                       # No boxes predicted
```

### Expected Ground Truth Format

```
[x1, y1, x2, y2]                # Single box
[x1, y1, x2, y2], [...]         # Multiple boxes
""  (empty string)              # No ground truth boxes
```

---

## Common Hyperparameter Guidance

### BETA (F-beta parameter)
- `0.5`: Emphasize precision (useful for medical, fewer false positives)
- `1.0`: Balanced F1 score (standard choice)
- `1.5`: Mild recall emphasis (don't miss findings)
- `2.0`: Strong recall emphasis (minimize missed detections)

**Medical Recommendation:** Start with 1.0, consider 0.5-1.0 for safety

### MIN_IOU_THRESHOLD
- `0.2`: Very lenient (accept rough boxes)
- `0.3`: Lenient (early training)
- `0.5`: Standard (COCO metric)
- `0.7`: Strict (require precise localization)

**Medical Recommendation:** 0.5-0.7 (precision important)

### NO_BOX_BONUS / NO_FINDINGS_REWARD
- Too high (>0.5): Model may exploit "no findings" predictions
- Too low (<0.1): Model unfairly penalized for correct negatives
- Sweet spot: 0.2-0.3

---

## Summary Table

| Function | Speed | Complexity | Stability | Multi-Box | Best For |
|----------|-------|-----------|-----------|-----------|----------|
| **R1** | Fast | Very Low | High | Moderate | Baseline/evaluation matching |
| **R2/R3** | Fast | Low | High | Good | General GRPO baseline |
| **R4** | Fast | Medium | High | Good | Advanced GRPO, medical safety |
| **R5** | Slowest | Highest | Medium | Best | Exploration, complex scenarios |

---

**Summary Generated:** October 2025
**Framework:** Master Thesis GPRO Reward Functions Analysis
**Status:** All reward functions fully documented and evaluated

