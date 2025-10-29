# Reward Functions for GRPO Training

Organized from **least to most complex** for radiology grounding tasks.

## Overview

This folder contains 5 reward functions designed for GRPO (Group Relative Policy Optimization) training of vision-language models on bounding box detection tasks. They are numbered by increasing complexity.

## Reward Functions (Ordered by Complexity)

### R1.py - Simple AP@0.5 Baseline ⭐
**Complexity**: Simplest
**Approach**: Binary threshold at IoU=0.5, computes Average Precision
**Hyperparameters**: 1
- `NO_BOX_BONUS = 0.2`

**Best for**:
- Baseline reference
- Matching evaluation metrics (AP@0.5)
- Understanding core concepts

**Key Formula**:
```
reward = AP@0.5 (binary: IoU ≥ 0.5 → match, else no match)
```

---

### R2.py - F-beta × IoU (RL-Optimized) ⭐⭐
**Complexity**: Simple
**Approach**: F_β score × mean_IoU with greedy matching
**Hyperparameters**: 3
- `BETA = 1.0` (F-beta parameter: 1.0=balanced, 1.5=mild recall, 2.0=strong recall)
- `MIN_IOU_THRESHOLD = 0.5` (matching threshold)
- `NO_BOX_BONUS = 0.2`

**Best for**:
- Clean GRPO baseline
- Well-balanced detection/localization
- Easy to understand and tune

**Key Formula**:
```
F_β = (1+β²) × P×R / (β²×P + R)
reward = F_β × mean_IoU
```

**Advantages**:
✅ Dense rewards (every IoU value contributes)
✅ Clear credit assignment (F_β = detection, mean_IoU = localization)
✅ Natural penalties (F-beta handles over/under-prediction)
✅ Bounded [0,1], stable for RL

---

### R3.py - Soft Partial Credit ⭐⭐½
**Complexity**: Simple-Medium
**Approach**: Lower threshold (0.2) with quadratic partial credit, weighted F-beta
**Hyperparameters**: 3
- `MIN_IOU_THRESHOLD = 0.2` (lowered to accept mediocre attempts)
- `BETA = 1.0`
- `NO_BOX_BONUS = 0.2`

**Best for**:
- Smooth GRPO training
- Giving partial credit for "almost correct" predictions
- Avoiding cliff effects at threshold

**Key Formula**:
```
quality(IoU) = {
    0.0                    if IoU < 0.2
    0.5 × ((IoU-0.2)/0.3)² if 0.2 ≤ IoU < 0.5  (partial credit!)
    IoU                    if IoU ≥ 0.5
}
reward = F_β (using quality-weighted TP)
```

**Advantages over R2**:
✅ Smooth learning signal (no cliff at 0.5)
✅ Partial credit for IoU=0.3-0.5 (not zero!)
✅ Encourages improvement (steeper gradient where it matters)

**Examples**:
- IoU=0.45 → quality=0.39 → contributes to reward! (R2 would give 0)
- IoU=0.35 → quality=0.19 → some credit

---

### R4.py - Enhanced Smooth Gradients ⭐⭐⭐
**Complexity**: Medium
**Approach**: F_β × smooth_quality using cubic spline transformation
**Hyperparameters**: 4-5
- `BETA = 1.5`
- `MIN_IOU_THRESHOLD = 0.5`
- `NO_BOX_BONUS = 0.2`
- `USE_CENTER_AWARE = False` (optional)
- `CENTER_WEIGHT = 0.15` (if center-aware enabled)

**Best for**:
- Advanced GRPO training
- Better gradient shaping
- Optional geometric refinement

**Key Formula**:
```
smooth_quality = mean(spline(IoU_i))
reward = F_β × smooth_quality

Where spline uses cubic Hermite interpolation with control points:
- (0.0, 0.0, gradient=1.5)  - steep initial gradient
- (0.5, 0.5, gradient=1.1)  - at threshold
- (0.8, 0.8, gradient=0.9)  - good zone
- (1.0, 1.0, gradient=0.5)  - diminishing returns
```

**Advantages over R2/R3**:
✅ Smoother gradients than linear mean_IoU
✅ Steeper gradient [0-0.5] encourages crossing threshold
✅ Gentler gradient [0.8-1.0] avoids perfectionism
✅ Optional center-aware penalty for geometric precision

**Requires**: `scipy` for spline interpolation

---

### R5.py - Strict Medical Focus ⭐⭐⭐⭐
**Complexity**: Most Complex
**Approach**: Hungarian matching, piecewise linear IoU reward, unified penalties
**Hyperparameters**: 5
- `BETA = 1.0` (can use β < 1 for precision focus)
- `IOU_THRESHOLD = 0.5`
- `PENALTY_STRENGTH = 0.7` (error harshness)
- `NO_FINDINGS_REWARD = 0.3`
- `RECALL_BONUS = 0.1` (complete detection bonus)

**Best for**:
- Medical applications requiring precision
- Strict quality requirements
- Minimal false positives

**Key Formula**:
```
IoU_reward(IoU) = {
    0                             if IoU < 0.15
    0.5 × (IoU-0.15)/(τ-0.15)    if 0.15 ≤ IoU < τ
    0.5 + 0.5 × (IoU-τ)/(1-τ)    if IoU ≥ τ
}

penalty(k) = exp(-α × k)  where k = number of errors

F_β with weighted TP, plus penalties and recall bonus
```

**Advantages**:
✅ Hungarian matching (globally optimal)
✅ Unified penalty system
✅ Recall bonus for complete detection
✅ Designed for medical precision (can use β=0.5)

**Use when**: False positives are costly (medical imaging)

---

## Recommendation for GRPO

### Quick Start
Start with **R2** - it's clean, well-documented, and works well out of the box.

### For Different Scenarios

| Scenario | Recommended | Why |
|----------|------------|-----|
| Baseline/Reference | **R1** | Matches AP@0.5 evaluation metric |
| General GRPO | **R2** | Clean, balanced, easy to tune |
| Smooth training | **R3** | Partial credit helps exploration |
| Advanced tuning | **R4** | Better gradient shaping |
| Medical imaging | **R5** | Strict, precision-focused |

### Progression Path
1. Start with **R2** (baseline GRPO)
2. If training is choppy → try **R3** (partial credit)
3. If training stagnates → try **R4** (better gradients)
4. For production medical → consider **R5** (strict)

---

## Common Interface

All reward functions implement:
```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute reward for grounding task.

    Args:
        data_source: Dataset name
        solution_str: Model output with <answer>[x1,y1,x2,y2]</answer>
        ground_truth: GT boxes as "[x1,y1,x2,y2],[x1,y1,x2,y2],..."
        extra_info: Optional metadata
        return_details: If True, return detailed metrics dict

    Returns:
        Reward score [0, 1] or detailed metrics dictionary
    """
```

## Testing

Each file includes test cases in `if __name__ == "__main__"` block:
```bash
python R1.py  # Test R1
python R2.py  # Test R2
# etc.
```

---

## Hyperparameter Tuning Guide

### BETA (F-beta parameter)
- `β = 0.5`: Emphasize precision (fewer false positives)
- `β = 1.0`: Balanced (F1 score)
- `β = 1.5`: Mild recall emphasis
- `β = 2.0`: Strong recall emphasis (fewer misses)

**Medical imaging**: Consider β ≤ 1.0 to minimize false positives

### IOU_THRESHOLD
- `0.3`: Lenient (accept rough boxes)
- `0.5`: Standard (COCO metric)
- `0.7`: Strict (require precise localization)

### NO_BOX_BONUS
- Too high (>0.5): Model may exploit "no findings" predictions
- Too low (<0.1): Model penalized for correct negatives
- Sweet spot: 0.2-0.3

---

## References

- **AP@0.5**: COCO detection metrics
- **F-beta**: Harmonic mean of precision and recall
- **Hungarian Algorithm**: Optimal bipartite matching (used in R5)
- **Greedy Matching**: Fast approximate matching (used in R1-R4)

---

**Last Updated**: October 2025
**Organized by complexity** for thesis clarity and GRPO experimentation.
