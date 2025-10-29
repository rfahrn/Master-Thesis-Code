# Quick Reference Guide: GPRO Reward Functions

## At a Glance

| Aspect | R1 | R2 | R3 | R4 | R5 |
|--------|----|----|----|----|-----|
| **File** | R1.py | R2.py | R3.py | R4.py | R5.py |
| **Approach** | Binary AP@0.5 | F-β × IoU | Dense F-β × IoU | Spline + F-β | Hungarian + F-β |
| **Hyperparams** | 1 | 3 | 3 | 4-5 | 5 |
| **Complexity** | Very Low | Low | Low | Medium | High |
| **Speed** | Fast | Fast | Fast | Fast | Slowest |
| **Stability** | Good | Good | Excellent | Good | Medium |
| **Multi-Box** | OK | Good | Good | Good | Best |
| **Recommended** | Baseline | General | Production | Medical | Complex |

---

## The 30-Second Summary

**R1 (AP@0.5):** Simple baseline matching COCO metrics. Binary threshold at IoU=0.5.

**R2/R3 (F-β × IoU):** Dense rewards, F-β handles detection, mean_IoU handles localization. Clean, fast, proven.

**R4 (Smooth Gradients):** R3 + cubic spline for better reward shaping. Steeper gradient near threshold, gentler at extremes.

**R5 (Hungarian):** Most complex. Uses globally optimal Hungarian matching. Piecewise linear rewards. Best for strict medical.

---

## Which Should I Use?

### For Your First Try
**→ R3** or **→ R4**

R3 is simpler (3 params), R4 is more advanced (4-5 params). Both use fast greedy matching. R4 has better gradient shaping.

### For Best Learning Speed
**→ R5**

Highest mean reward (0.4935). Partial credit mechanism helps early training.

### For Maximum Stability
**→ R3** or **→ R6**

Lowest variance (0.3691). Most reproducible. Best for thesis.

### For Medical Imaging
**→ R4**

Most strict on false negatives (reward=0.2653 vs 0.3333 for others). Good safety profile.

### For Production/Deployment
**→ R1** (if matching COCO)
**→ R3** (if general training)
**→ R5** (if zero false positives critical)

---

## Common Utilities (All Functions Use These)

```python
# Extract bounding boxes from prediction text
extract_bounding_boxes(answer: str) -> List[List[float]]

# Calculate IoU between two boxes
compute_iou(box1: List[float], box2: List[float]) -> float

# Classify edge cases (some functions)
classify_edge_case(n_pred: int, n_gt: int) -> str
# Returns: "true_negative", "hallucination", "missed_detection", "one_to_one", etc.

# Greedy matching (R1-R4)
greedy_match_boxes(pred_boxes, actual_boxes, min_iou) -> Tuple[matches, iou_matrix]

# Hungarian matching (R5)
hungarian_matching(pred_boxes, gt_boxes) -> Tuple[iou_values, num_matches]
```

---

## Mathematical Formulas (Quick Ref)

### R1: AP@0.5
```
reward = AP@0.5 (binary IoU threshold)
```

### R2/R3/R4: F-beta × Quality
```
reward = F_β × quality
where F_β = (1+β²) × (P×R) / (β²×P + R)
      P = matches / predictions
      R = matches / ground_truth
```

### R4: Smooth Quality (Special)
```
quality = mean(spline(IoU_i))
# Spline control points: (0,0,1.5), (0.5,0.5,1.1), (0.8,0.8,0.9), (1,1,0.5)
```

### R5: Hungarian + Piecewise
```
matches from Hungarian algorithm
quality(IoU) = piecewise linear with 3 regions
penalty(k) = exp(-strength × k)
```

---

## Hyperparameters by Function

### R1
- `NO_BOX_BONUS = 0.2`

### R2
- `BETA = 1.0` (balanced F1)
- `MIN_IOU_THRESHOLD = 0.5` (COCO standard)
- `NO_BOX_BONUS = 0.2`

### R3 (Same as R2)
- `BETA = 1.0`
- `MIN_IOU_THRESHOLD = 0.5`
- `NO_BOX_BONUS = 0.2`

### R4
- `BETA = 1.5` (mild recall for medical)
- `MIN_IOU_THRESHOLD = 0.5`
- `NO_BOX_BONUS = 0.2`
- `USE_CENTER_AWARE = False` (optional)
- `CENTER_WEIGHT = 0.15` (if enabled)

### R5
- `BETA = 1.0` (use 0.5 for precision emphasis)
- `IOU_THRESHOLD = 0.5`
- `PENALTY_STRENGTH = 0.7` (0=gentle, 1=harsh)
- `NO_FINDINGS_REWARD = 0.3`
- `RECALL_BONUS = 0.1`

---

## Tuning Beta Parameter

```
BETA Value    Effect                          When to Use
0.5           2x weight on precision          Medical (minimize FP)
1.0           Balanced F1 score               General purpose (default)
1.5           Mild recall emphasis            Don't miss findings
2.0           4x weight on recall             Critical to find all
```

---

## Tuning MIN_IOU_THRESHOLD

```
Value    Harshness              When to Use
0.2      Very lenient           Early training, exploration
0.3      Lenient                Warm-up phase
0.5      Standard (COCO)        General (default)
0.7      Strict                 Fine-tuning, medical
```

---

## Expected Performance Ranges

### Perfect Match (IoU=1.0, all matched)
- R1: 1.0
- R2/R3: 1.0
- R4: 1.0
- R5: 1.0

### Good Match (IoU≈0.7, all matched)
- R1: 0.81
- R2/R3: 0.70
- R4: 0.70 (with spline)
- R5: 0.65+

### Partial Match (50% detected, IoU≈0.6 avg)
- R1: 0.33
- R2/R3: 0.40
- R4: 0.42
- R5: 0.50+ (better!)

### True Negative (both empty)
- All: 0.2 (NO_BOX_BONUS)

---

## Standard Interface (All Functions)

```python
def compute_score(
    data_source: str,           # Dataset name
    solution_str: str,          # Model output: "<answer>[x1,y1,x2,y2]</answer>"
    ground_truth: str,          # GT boxes: "[x1,y1,x2,y2],[...]"
    extra_info=None,            # Optional metadata
    return_details: bool=False  # Get detailed metrics?
) -> float | Dict[str, Any]:
    pass

# Usage
from Reward_Functions import R4
reward = R4.compute_score("dataset_name", "[0.1,0.2,0.3,0.4]", "[0.1,0.2,0.3,0.4]")
```

---

## Input/Output Formats

### Model Output Expected
```
<answer>[x1, y1, x2, y2]</answer>          # Single box
<answer>[x1, y1, x2, y2], [...]</answer>   # Multiple boxes
<answer></answer>                          # No boxes (empty prediction)
```

### Ground Truth Format
```
[x1, y1, x2, y2]              # Single box
[x1, y1, x2, y2], [...]       # Multiple boxes
""  (empty string)            # No boxes
```

### Return Values (return_details=False)
```
float: Reward in [0, 1]
Typical values:
- 0.0: Wrong answer (hallucination, missed detection)
- 0.2: True negative (both empty)
- 0.3-0.7: Partial credit
- 0.8-1.0: Good/perfect match
```

### Return Values (return_details=True)
```
Dict with keys:
- 'reward': float (main score)
- 'precision': float (detection quality)
- 'recall': float (coverage)
- 'mean_iou': float (localization quality)
- 'fbeta_score': float (F-beta)
- 'matches': List of (pred_idx, gt_idx, iou)
- 'num_matches': int
- 'num_predictions': int
- 'num_ground_truth': int
- 'edge_case': str (true_negative, hallucination, etc.)
- 'iou_matrix': numpy array
```

---

## Edge Cases (How Each Function Handles)

| Scenario | R1 | R2/R3 | R4 | R5 |
|----------|----|----|----|----|
| No pred, No GT | 0.2 | 0.2 | 0.2 | 0.3 |
| Pred exist, No GT | 0.0 | 0.0 | 0.0 | Harsh penalty |
| No pred, GT exist | 0.0 | 0.0 | 0.0 | 0.0 |
| 1/2 matched, IoU=0.6 | 0.33 | 0.40 | 0.42 | 0.50+ |
| 10 pred, 1 GT (bad) | 0.10 | 0.33 | 0.45 | 0.20 |

---

## Testing Each Function

```bash
# Test R1
python Reward_Functions/R1.py

# Test R2/R3/R4
python Reward_Functions/R3.py

# Test R4 specifically
python Reward_Functions/R4.py

# Test R5
python Reward_Functions/R5.py

# Evaluate all together
python evaluate_reward_functions.py
```

---

## Curriculum Learning Strategy (Advanced)

**Phase 1: Warm-up (0-20% of training)**
```python
USE R5 with MIN_IOU=0.2
# Goals: Learn basic detection, accept imperfect attempts
# Strategy: Partial credit helps exploration
```

**Phase 2: Refinement (20-70% of training)**
```python
USE R4 with MIN_IOU=0.5
# Goals: Improve precision, localization
# Strategy: Smooth gradients stabilize learning
```

**Phase 3: Fine-tuning (70-100% of training)**
```python
USE R3 with MIN_IOU=0.5
# Goals: Optimize for final metrics
# Strategy: Most stable, best reproducibility
```

---

## File Locations

```
/home/user/Master-Thesis-Code/
├── Reward_Functions/
│   ├── R1.py
│   ├── R2.py
│   ├── R3.py
│   ├── R4.py
│   ├── R5.py (same as R7 Simplified)
│   └── README.md (comprehensive guide)
├── REWARD_FUNCTIONS_DETAILED_SUMMARY.md (full reference)
├── QUICK_REFERENCE_GUIDE.md (this file)
├── REWARD_FUNCTION_ANALYSIS.md (evaluation results)
├── evaluate_reward_functions.py (comparison framework)
├── test_r1_simple.py (simple test)
└── grpo_simulation.py (integration example)
```

---

## Key Insights

1. **R3 is the "Goldilocks" choice** - Simple (3 params), fast, proven
2. **R4 is better for medical** - Strict on false negatives
3. **R5 is best for exploration** - Partial credit helps learning
4. **All use same interface** - Easy to swap between functions
5. **Greedy matching is 15x faster** - Than Hungarian, usually same quality

---

## Common Mistakes

❌ **Don't:** Use IoU=0.7 threshold. Standard is 0.5 (COCO)
✅ **Do:** Start with MIN_IOU_THRESHOLD = 0.5

❌ **Don't:** Set NO_BOX_BONUS > 0.5. Encourages exploitation
✅ **Do:** Use 0.2-0.3 range

❌ **Don't:** Use β=2.0 for general cases. Only if missing findings critical
✅ **Do:** Start with β=1.0 (balanced)

❌ **Don't:** Mix greedy and Hungarian across training. Inconsistent
✅ **Do:** Stick with R3/R4 (greedy) or R5 (Hungarian) throughout

---

## For Thesis Writing

**Recommended Statement:**
> "We evaluated reward functions R3 (F-β × mean_IoU), R4 (Enhanced Smooth Gradients), and R5 (Strict Medical Focus) for GRPO-based VLM training on radiology grounding. R4 was selected as the primary choice due to its superior gradient smoothness and appropriate safety profile for medical applications."

**Key Metrics to Report:**
- Mean reward across test scenarios
- Standard deviation (stability)
- Performance on complex multi-box scenarios
- False negative handling (safety)

---

**Last Updated:** October 2025
**Format:** Quick Reference (keep in tab while coding)
**Full Guide:** See REWARD_FUNCTIONS_DETAILED_SUMMARY.md

