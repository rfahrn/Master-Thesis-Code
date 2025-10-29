"""
R7 Simplified: Strict Medical Grounding with Minimal Hyperparameters

Simplified to just 5 core parameters for clear thesis explanation:
1. BETA - F-beta parameter (0.5 = precision focus, 2.0 = recall focus)
2. IOU_THRESHOLD - Minimum IoU for reward (strict = 0.5)
3. PENALTY_STRENGTH - How harsh to penalize errors (0-1)
4. NO_FINDINGS_REWARD - Score for correct negatives
5. RECALL_BONUS - Optional bonus for finding all boxes

Mathematical formulation remains powerful but simpler to explain.
"""

import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==================== SIMPLIFIED CONFIGURATION ====================
# Only 5 core hyperparameters for thesis clarity

# Core detection parameters
BETA = 1                    # F-beta: 0.5=precision focus, 1.0=balanced, 2.0=recall focus
IOU_THRESHOLD = 0.5           # Minimum IoU for "good" detection (strict for medical)
PENALTY_STRENGTH = 0.7        # Penalty harshness (0=gentle, 1=harsh)

# Reward structure
NO_FINDINGS_REWARD = 0.3      # Reward for correct no-findings (low to prevent exploitation)
RECALL_BONUS = 0.1            # Bonus for finding ALL boxes (set 0 to disable)

# Fixed internal parameters (not exposed as hyperparameters)
MIN_BOX_SIZE = 0.001          # Minimum normalized box size
MIN_IOU_FOR_REWARD = 0.15     # Below this IoU = zero reward (fixed)


# ==================== SIMPLIFIED REWARD FUNCTIONS ====================

def simple_iou_reward(iou: float, threshold: float = 0.5) -> float:
    """
    Simplified IoU reward with three clear regions.
    
    Mathematical formula:
    - IoU < 0.15: reward = 0 (too poor)
    - 0.15 ≤ IoU < threshold: reward = 0.5 × (IoU - 0.15) / (threshold - 0.15)
    - IoU ≥ threshold: reward = 0.5 + 0.5 × (IoU - threshold) / (1 - threshold)
    
    This creates a piecewise linear function with clear interpretation.
    """
    if iou < MIN_IOU_FOR_REWARD:
        return 0.0
    elif iou < threshold:
        # Linear ramp to 0.5 at threshold
        range_size = threshold - MIN_IOU_FOR_REWARD
        return 0.5 * (iou - MIN_IOU_FOR_REWARD) / range_size
    else:
        # Linear from 0.5 to 1.0
        range_size = 1.0 - threshold
        return 0.5 + 0.5 * (iou - threshold) / range_size


def compute_penalty(num_errors: int, strength: float = 0.7) -> float:
    """
    Unified penalty function for false positives and overprediction.
    
    Formula: penalty = exp(-strength × num_errors)
    
    Examples with strength=0.7:
    - 1 error: 0.50
    - 2 errors: 0.25
    - 3 errors: 0.12
    """
    return np.exp(-strength * num_errors)


# ==================== CORE FUNCTIONS ====================

def extract_boxes(text: str) -> List[List[float]]:
    """Extract normalized [0,1] bounding boxes from text."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"
    
    boxes = []
    for match in re.finditer(pattern, text):
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            
            # Validate normalized coordinates
            if not all(np.isfinite(coords)):
                continue
            if not all(-0.01 <= c <= 1.01 for c in coords):
                continue
                
            # Clip and order
            x1, y1, x2, y2 = [np.clip(c, 0.0, 1.0) for c in coords]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Check minimum size
            if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                boxes.append([x1, y1, x2, y2])
                
        except (ValueError, IndexError):
            continue
            
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def hungarian_matching(pred_boxes: List[List[float]], 
                       gt_boxes: List[List[float]]) -> Tuple[List[float], int]:
    """
    Perform Hungarian matching and return IoU values and match count.
    Simplified to return just what we need.
    """
    if not pred_boxes or not gt_boxes:
        return [], 0
        
    M, N = len(pred_boxes), len(gt_boxes)
    
    # Build IoU matrix
    iou_matrix = np.zeros((M, N))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    # Hungarian algorithm
    cost_matrix = 1.0 - iou_matrix
    
    if M != N:
        max_dim = max(M, N)
        padded_cost = np.ones((max_dim, max_dim))
        padded_cost[:M, :N] = cost_matrix
        row_ind, col_ind = linear_sum_assignment(padded_cost)
        
        iou_values = []
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                iou_values.append(float(iou_matrix[r, c]))
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        iou_values = [float(iou_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
        
    return iou_values, len(iou_values)


# ==================== MAIN SIMPLIFIED REWARD ====================

def compute_r7_simplified(pred_boxes: List[List[float]],
                         gt_boxes: List[List[float]],
                         beta: float = BETA,
                         iou_threshold: float = IOU_THRESHOLD,
                         penalty_strength: float = PENALTY_STRENGTH) -> Dict:
    """
    Simplified R7 reward computation with minimal parameters.
    
    Core formula:
    1. Match boxes using Hungarian algorithm
    2. Compute F-beta score with IoU-weighted TP
    3. Apply penalties for false positives or overprediction
    4. Add recall bonus if all boxes found
    
    Args:
        pred_boxes: Predicted boxes
        gt_boxes: Ground truth boxes  
        beta: F-beta parameter (0.5=precision, 2.0=recall)
        iou_threshold: Minimum IoU for good detection
        penalty_strength: How harsh penalties are
        
    Returns:
        Dictionary with score and components
    """
    M = len(pred_boxes)  # Number predicted
    N = len(gt_boxes)    # Number ground truth
    
    # Initialize result
    result = {
        'score': 0.0,
        'num_pred': M,
        'num_gt': N,
        'precision': 0.0,
        'recall': 0.0,
        'avg_iou': 0.0,
        'case': 'unknown'
    }
    
    # === CASE 1: No ground truth (no findings expected) ===
    if N == 0:
        if M == 0:
            # Correct no-findings
            result['score'] = NO_FINDINGS_REWARD
            result['case'] = 'correct_negative'
        else:
            # False positives - apply penalty
            penalty = compute_penalty(M, penalty_strength)
            result['score'] = NO_FINDINGS_REWARD * penalty * 0.5  # Extra harsh for FP
            result['case'] = f'false_positive_{M}'
        return result
    
    # === CASE 2: Missed all findings ===
    if M == 0:
        result['score'] = 0.0
        result['case'] = 'missed_all'
        return result
    
    # === CASE 3: Normal detection - compute F-beta ===
    
    # Get IoU values from optimal matching
    iou_values, num_matches = hungarian_matching(pred_boxes, gt_boxes)
    
    # Compute weighted true positives
    tp_weighted = sum(simple_iou_reward(iou, iou_threshold) for iou in iou_values)
    
    # Compute precision and recall
    precision = tp_weighted / M if M > 0 else 0.0
    recall = tp_weighted / N if N > 0 else 0.0
    
    # F-beta score
    if precision + recall > 0:
        beta_sq = beta * beta
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    else:
        f_beta = 0.0
    
    # Apply overprediction penalty if too many boxes
    if M > N * 1.5:  # More than 50% extra boxes
        excess = M - N
        penalty = compute_penalty(excess, penalty_strength * 0.5)  # Gentler than FP
        f_beta *= penalty
    
    # Apply recall bonus if all boxes found with good IoU
    if RECALL_BONUS > 0 and N >= 2 and num_matches == N:
        if all(iou >= iou_threshold for iou in iou_values[:N]):
            f_beta = min(1.0, f_beta + RECALL_BONUS)
            result['recall_bonus'] = True
    
    # Store results
    result['score'] = float(np.clip(f_beta, 0.0, 1.0))
    result['precision'] = float(precision)
    result['recall'] = float(recall)
    result['avg_iou'] = float(np.mean(iou_values)) if iou_values else 0.0
    result['case'] = 'detection'
    
    return result


# ==================== VERL INTERFACE ====================

def compute_score(data_source: str, solution_str: str, ground_truth: str,
                 extra_info: Optional[Dict] = None) -> float:
    """
    VeRL interface - simplified with optional parameter override.
    
    Extra info can contain:
    - beta: Override F-beta parameter
    - iou_threshold: Override IoU threshold
    - penalty_strength: Override penalty strength
    """
    # Extract boxes
    pred_boxes = extract_boxes(solution_str)
    gt_boxes = extract_boxes(ground_truth)
    
    # Get parameters (use defaults or override from extra_info)
    beta = BETA
    iou_threshold = IOU_THRESHOLD
    penalty_strength = PENALTY_STRENGTH
    
    if extra_info:
        beta = extra_info.get('beta', beta)
        iou_threshold = extra_info.get('iou_threshold', iou_threshold)
        penalty_strength = extra_info.get('penalty_strength', penalty_strength)
    
    # Compute score
    result = compute_r7_simplified(
        pred_boxes, gt_boxes, 
        beta, iou_threshold, penalty_strength
    )
    
    return result['score']


# ==================== ANALYSIS & TESTING ====================

def analyze_reward(solution_str: str, ground_truth: str,
                  beta: float = BETA,
                  iou_threshold: float = IOU_THRESHOLD) -> Dict:
    """Detailed analysis for debugging."""
    pred_boxes = extract_boxes(solution_str)
    gt_boxes = extract_boxes(ground_truth)
    
    return compute_r7_simplified(
        pred_boxes, gt_boxes,
        beta, iou_threshold, PENALTY_STRENGTH
    )


if __name__ == "__main__":
    print("=" * 80)
    print("R7 SIMPLIFIED - MINIMAL HYPERPARAMETERS FOR THESIS")
    print("=" * 80)
    print(f"\nOnly 5 Core Parameters:")
    print(f"  1. BETA = {BETA} (F-beta: precision vs recall)")
    print(f"  2. IOU_THRESHOLD = {IOU_THRESHOLD} (minimum for good detection)")
    print(f"  3. PENALTY_STRENGTH = {PENALTY_STRENGTH} (error harshness)")
    print(f"  4. NO_FINDINGS_REWARD = {NO_FINDINGS_REWARD} (correct negative score)")
    print(f"  5. RECALL_BONUS = {RECALL_BONUS} (complete detection bonus)")
    
    # Test cases
    test_cases = [
        ("Perfect", "[0.3, 0.4, 0.5, 0.6]", "[0.3, 0.4, 0.5, 0.6]"),
        ("Good (70% IoU)", "[0.30, 0.40, 0.48, 0.58]", "[0.30, 0.40, 0.50, 0.60]"),
        ("Threshold (50% IoU)", "[0.30, 0.40, 0.45, 0.55]", "[0.30, 0.40, 0.50, 0.60]"),
        ("Poor (30% IoU)", "[0.25, 0.35, 0.40, 0.50]", "[0.30, 0.40, 0.50, 0.60]"),
        ("Very Poor (10% IoU)", "[0.1, 0.1, 0.2, 0.2]", "[0.3, 0.4, 0.5, 0.6]"),
        ("Correct negative", "", ""),
        ("1 False positive", "[0.1, 0.1, 0.2, 0.2]", ""),
        ("2 False positives", "[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]", ""),
        ("Missed all", "", "[0.3, 0.4, 0.5, 0.6]"),
        ("2 perfect (bonus)", "[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]", 
         "[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]"),
        ("Overprediction", "[0.3,0.4,0.5,0.6], [0.1,0.1,0.2,0.2], [0.7,0.7,0.8,0.8]", 
         "[0.3, 0.4, 0.5, 0.6]"),
    ]
    
    print("\n" + "-" * 80)
    print("TEST RESULTS:")
    print("-" * 80)
    
    for name, pred, gt in test_cases:
        result = analyze_reward(pred, gt)
        print(f"\n{name}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Case: {result['case']}")
        if result['avg_iou'] > 0:
            print(f"  Avg IoU: {result['avg_iou']:.3f}")
            print(f"  P/R: {result['precision']:.2f}/{result['recall']:.2f}")
        if result.get('recall_bonus'):
            print(f"  Recall bonus applied!")
    
    # Mathematical explanation
    print("\n" + "=" * 80)
    print("MATHEMATICAL FORMULATION FOR THESIS:")
    print("=" * 80)
    print("""
1. F-β Score:
   F_β = (1 + β²) × P × R / (β² × P + R)
   
   Where P = Σ(IoU_reward) / M  (weighted precision)
         R = Σ(IoU_reward) / N  (weighted recall)

2. IoU Reward (piecewise linear):
   reward(IoU) = {
       0                             if IoU < 0.15
       0.5 × (IoU-0.15)/(τ-0.15)    if 0.15 ≤ IoU < τ
       0.5 + 0.5 × (IoU-τ)/(1-τ)    if IoU ≥ τ
   }
   Where τ = IOU_THRESHOLD

3. Penalty Function:
   penalty(k) = exp(-α × k)
   Where α = PENALTY_STRENGTH, k = number of errors

4. Final Score:
   - No findings: score = NO_FINDINGS_REWARD × penalty(M) × 0.5
   - Detection: score = F_β × overpred_penalty + recall_bonus
   
5. Parameters (only 5!):
   β ∈ [0.5, 2.0]: precision-recall trade-off
   τ ∈ [0.3, 0.7]: IoU threshold
   α ∈ [0.3, 1.0]: penalty strength
   r_nf ∈ [0.2, 0.5]: no-findings reward
   r_b ∈ [0, 0.2]: recall bonus
""")
    
    print("\n" + "=" * 80)
    print("THESIS EXPLANATION:")
    print("=" * 80)
    print("""
"R7 uses just 5 hyperparameters to create a strict reward function
suitable for medical grounding tasks:

1. β controls the precision-recall trade-off in the F-beta score
2. τ sets the IoU threshold for acceptable detections  
3. α determines how harshly errors are penalized
4. r_nf prevents exploitation of no-findings predictions
5. r_b optionally rewards complete detection

The key insight is using β=0.5 to emphasize precision over recall,
critical for medical applications where false positives are costly."
""")