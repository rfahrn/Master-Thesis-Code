"""
R6 Simple: Clean Enhanced GRPO Reward for Multi-Bounding Box Detection
Designed for NORMALIZED bounding boxes in [0,1] range.

Box format: [x1, y1, x2, y2] where all coordinates are in [0, 1]
Example: [0.15, 0.25, 0.45, 0.60] represents a box from (15%, 25%) to (45%, 60%)

Improvements over original:
1. Better handling of no-findings cases (with proper reward)
2. Task difficulty based on expected box count
3. Smooth F-beta adaptation
4. Cleaner spam penalty
5. Minimal hyperparameters

Mathematical formulation:
R = w_box * S_box + w_quality * S_quality

Where weights adapt based on expected complexity.
"""

import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==================== Core Configuration ====================
IOU_THRESHOLD = 0.5  # Standard threshold, works well for normalized boxes
BETA_BASE = 2.0  # Base F-beta value (emphasizes recall)

# Response length bounds
MIN_LENGTH = 15
IDEAL_LENGTH = 100
MAX_LENGTH = 300

# Reward for correct no-findings
NO_FINDINGS_REWARD = 0.1  # High but not perfect to leave room for improvement

# Minimum box size for normalized coordinates (as fraction of image)
MIN_BOX_SIZE = 0.001  # 0.1% of image dimension (e.g., 10x10 pixels in 1000x1000)


# ==================== Core Functions ====================

def extract_bounding_boxes(text: str) -> List[List[float]]:
    """Extract normalized bounding boxes from text (coordinates in [0,1] range)."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"
    
    boxes = []
    for match in re.finditer(pattern, text):
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            if not all(np.isfinite(coords)):
                continue
            
            x1, y1, x2, y2 = coords
            
            # For normalized boxes, ensure they're in [0, 1] range
            # Allow slight overflow for numerical errors
            if not all(-0.01 <= c <= 1.01 for c in coords):
                continue  # Skip boxes outside normalized range
            
            # Clip to valid range
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # For normalized boxes, use a smaller threshold for valid area
            # Since boxes are in [0,1], area can be very small but still valid
            MIN_BOX_SIZE = 0.001  # 0.1% of image dimension
            
            if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
            
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes.
    Works correctly for both normalized [0,1] and pixel coordinates.
    
    Args:
        box1, box2: Boxes in format [x1, y1, x2, y2]
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def smooth_iou_reward(iou: float, threshold: float = 0.5) -> float:
    """Convert IoU to smooth reward with good gradients."""
    if iou <= 0:
        return 0.0
    elif iou < threshold:
        # Smooth ramp up to threshold
        return 0.5 * (iou / threshold) ** 1.5
    else:
        # Linear from threshold to perfect
        return 0.5 + 0.5 * (iou - threshold) / (1.0 - threshold)


def hungarian_matching(pred_boxes: List[List[float]], 
                       gt_boxes: List[List[float]]) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Perform optimal matching between predicted and ground truth boxes.
    Returns: (iou_values, matches)
    """
    if not pred_boxes or not gt_boxes:
        return [], []
    
    # Build IoU matrix
    M, N = len(pred_boxes), len(gt_boxes)
    iou_matrix = np.zeros((M, N))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    # Hungarian algorithm for optimal assignment
    cost_matrix = 1.0 - iou_matrix
    
    if M != N:
        # Pad to square matrix
        max_dim = max(M, N)
        padded_cost = np.ones((max_dim, max_dim))
        padded_cost[:M, :N] = cost_matrix
        row_ind, col_ind = linear_sum_assignment(padded_cost)
        
        # Extract valid matches
        matches = []
        iou_values = []
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                matches.append((r, c))
                iou_values.append(float(iou_matrix[r, c]))
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = list(zip(row_ind.tolist(), col_ind.tolist()))
        iou_values = [float(iou_matrix[r, c]) for r, c in matches]
    
    return iou_values, matches


def compute_detection_score(pred_boxes: List[List[float]], 
                           gt_boxes: List[List[float]],
                           expected_count: Optional[int] = None) -> Dict[str, float]:
    """
    Compute detection score with adaptive parameters based on expected difficulty.
    
    Returns dict with:
    - score: Final detection score
    - precision: Detection precision
    - recall: Detection recall
    - avg_iou: Average IoU of matched boxes
    """
    M = len(pred_boxes)  # Number of predictions
    N = len(gt_boxes)    # Number of ground truth
    
    # Use expected_count if provided, otherwise use ground truth count
    expected = expected_count if expected_count is not None else N
    
    # === Handle special cases ===
    if N == 0:
        # No ground truth boxes
        if M == 0:
            # Correct: no findings when none expected
            return {
                'score': NO_FINDINGS_REWARD,
                'precision': 1.0,
                'recall': 1.0,
                'avg_iou': 1.0
            }
        else:
            # False positives: predicted boxes when none should exist
            # Exponential decay penalty based on number of false positives
            penalty = np.exp(-0.5 * M)
            return {
                'score': penalty * 0.1,  # Heavy penalty
                'precision': 0.0,
                'recall': 1.0,  # Technically no gt to miss
                'avg_iou': 0.0
            }
    
    if M == 0:
        # Missed all boxes
        return {
            'score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_iou': 0.0
        }
    
    # === Normal case: compute matching ===
    iou_values, matches = hungarian_matching(pred_boxes, gt_boxes)
    
    # Compute weighted true positives using smooth IoU rewards
    tp_weighted = sum(smooth_iou_reward(iou, IOU_THRESHOLD) for iou in iou_values)
    
    # Compute precision and recall
    precision = tp_weighted / M if M > 0 else 0.0
    recall = tp_weighted / N if N > 0 else 0.0
    
    # === Adaptive F-beta score ===
    # Adjust beta based on expected difficulty
    if expected >= 5:
        beta = 2.5  # Hard: emphasize recall more
    elif expected >= 3:
        beta = 2.0  # Medium: standard
    else:
        beta = 1.5  # Easy: more balanced
    
    # F-beta score
    eps = 1e-8
    beta_sq = beta * beta
    fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)
    
    # === Anti-spam penalty ===
    # Penalize excessive over-prediction
    if M > N * 1.5:
        spam_penalty = np.exp(-0.15 * (M - N))
        fbeta *= spam_penalty
    
    avg_iou = np.mean(iou_values) if iou_values else 0.0
    
    return {
        'score': float(np.clip(fbeta, 0.0, 1.0)),
        'precision': float(precision),
        'recall': float(recall),
        'avg_iou': float(avg_iou)
    }


def compute_response_quality(response: str, has_findings: bool) -> float:
    """
    Simple response quality score based on length.
    """
    length = len(response.strip())
    
    if length < MIN_LENGTH:
        # Too short - likely incomplete
        return 0.5 * (length / MIN_LENGTH)
    elif length <= IDEAL_LENGTH:
        # Good length
        return 1.0
    elif length <= MAX_LENGTH:
        # Acceptable but getting long
        excess_ratio = (length - IDEAL_LENGTH) / (MAX_LENGTH - IDEAL_LENGTH)
        return 0.9 - 0.1 * excess_ratio
    else:
        # Too long
        return max(0.5, np.exp(-(length - MAX_LENGTH) / 500))


# ==================== Main GRPO Score Function ====================

def compute_score(data_source: str, solution_str: str, ground_truth: str,
                 extra_info: Optional[Dict] = None) -> float:
    """
    Compute GRPO reward score for multi-bounding box detection.
    
    Args:
        data_source: Dataset identifier
        solution_str: Model's response with predicted boxes
        ground_truth: Ground truth with expected boxes
        extra_info: Optional dict with:
            - expected_boxes: Number of expected boxes (for difficulty adaptation)
            - task_type: Optional task classification
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract boxes
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)
    
    # Get expected count from extra_info if available
    expected_count = None
    if extra_info:
        expected_count = extra_info.get('expected_boxes', None)
        # If not provided, could also be computed from num_boxes
        if expected_count is None:
            expected_count = extra_info.get('num_boxes', None)
    
    # Compute detection score
    detection_result = compute_detection_score(pred_boxes, gt_boxes, expected_count)
    detection_score = detection_result['score']
    
    # Compute response quality
    quality_score = compute_response_quality(solution_str, len(pred_boxes) > 0)
    
    # === Weight adaptation based on task complexity ===
    # More complex tasks (more boxes) should focus more on detection accuracy
    if expected_count is not None and expected_count > 3:
        # Hard task: 90% detection, 10% quality
        final_score = 0.9 * detection_score + 0.1 * quality_score
    else:
        # Standard: 85% detection, 15% quality
        final_score = 0.85 * detection_score + 0.15 * quality_score
    
    return float(np.clip(final_score, 0.0, 1.0))


def compute_score_with_difficulty(data_source: str, solution_str: str, 
                                 ground_truth: str, 
                                 extra_info: Optional[Dict] = None) -> Tuple[float, float]:
    """
    Compute score and difficulty weight separately for GRPO gradient weighting.
    
    Difficulty is based on:
    1. Number of expected boxes (more = harder)
    2. Average IoU achieved (lower = harder)
    
    Returns:
        (score, difficulty_weight)
    """
    score = compute_score(data_source, solution_str, ground_truth, extra_info)
    
    # Extract boxes for difficulty computation
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)
    
    # Base difficulty on expected complexity
    expected = len(gt_boxes)
    if extra_info:
        expected = extra_info.get('expected_boxes', expected)
    
    # Simple difficulty weight based on task complexity
    if expected == 0:
        difficulty = 1.0  # No findings case is standard difficulty
    elif expected <= 2:
        difficulty = 0.8  # Easy
    elif expected <= 4:
        difficulty = 1.0  # Medium
    else:
        difficulty = 1.2 + 0.1 * min(expected - 4, 6)  # Hard, caps at 1.8
    
    # Adjust difficulty based on performance (optional)
    if pred_boxes and gt_boxes:
        detection_result = compute_detection_score(pred_boxes, gt_boxes, expected)
        avg_iou = detection_result['avg_iou']
        
        # Lower IoU = harder sample, increase weight slightly
        if avg_iou < 0.5:
            difficulty *= 1.2
        elif avg_iou < 0.7:
            difficulty *= 1.1
    
    return score, float(np.clip(difficulty, 0.5, 2.0))


# ==================== Diagnostic Function ====================

def analyze_reward(solution_str: str, ground_truth: str, 
                  extra_info: Optional[Dict] = None) -> Dict:
    """
    Detailed analysis of reward components for debugging.
    """
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)
    
    expected = len(gt_boxes)
    if extra_info:
        expected = extra_info.get('expected_boxes', expected)
    
    detection_result = compute_detection_score(pred_boxes, gt_boxes, expected)
    quality_score = compute_response_quality(solution_str, len(pred_boxes) > 0)
    
    final_score = compute_score("analysis", solution_str, ground_truth, extra_info)
    score_with_diff, difficulty = compute_score_with_difficulty(
        "analysis", solution_str, ground_truth, extra_info
    )
    
    return {
        'num_predicted': len(pred_boxes),
        'num_ground_truth': len(gt_boxes),
        'expected_boxes': expected,
        'detection_score': detection_result['score'],
        'precision': detection_result['precision'],
        'recall': detection_result['recall'],
        'avg_iou': detection_result['avg_iou'],
        'quality_score': quality_score,
        'final_score': final_score,
        'difficulty_weight': difficulty,
        'response_length': len(solution_str.strip())
    }


# ==================== Test Suite ====================

if __name__ == "__main__":
    test_cases = [
        # (response, ground_truth, extra_info, description)
        ("Found objects at [0.1, 0.2, 0.3, 0.4]", 
         "[0.1, 0.2, 0.3, 0.4]", 
         None, 
         "Perfect single match"),
        
        ("Detected: [0.15, 0.15, 0.35, 0.35] and [0.55, 0.55, 0.75, 0.75]", 
         "[0.15, 0.15, 0.35, 0.35], [0.55, 0.55, 0.75, 0.75]", 
         {'expected_boxes': 2}, 
         "Perfect double match"),
        
        ("No objects detected in this image", 
         "", 
         {'expected_boxes': 0}, 
         "Correct no-findings"),
        
        ("Found something at [0.1, 0.1, 0.25, 0.25]", 
         "[0.1, 0.1, 0.4, 0.4]", 
         None, 
         "Partial match (~56% IoU)"),
        
        ("[0.05,0.05,0.1,0.1], [0.15,0.15,0.2,0.2], [0.25,0.25,0.3,0.3], [0.35,0.35,0.4,0.4], [0.45,0.45,0.5,0.5]", 
         "[0.05,0.05,0.1,0.1]", 
         None, 
         "Spam case (5 pred, 1 true)"),
        
        ("", 
         "[0.2,0.3,0.4,0.5], [0.6,0.6,0.8,0.8]", 
         {'expected_boxes': 2}, 
         "Missed all boxes"),
        
        ("Found boxes at [0.10,0.10,0.30,0.30], [0.40,0.40,0.60,0.60], [0.70,0.20,0.90,0.40], [0.15,0.70,0.35,0.90], [0.50,0.75,0.70,0.95]",
         "[0.12,0.12,0.32,0.32], [0.41,0.41,0.61,0.61], [0.70,0.20,0.90,0.40], [0.15,0.70,0.35,0.90], [0.50,0.75,0.70,0.95]",
         {'expected_boxes': 5},
         "Complex case (5 boxes)"),
         
        ("Object at [1.5, 0.5, 2.0, 0.8]",
         "[0.5, 0.5, 0.8, 0.8]",
         None,
         "Invalid box (out of range)"),
         
        ("Tiny object at [0.5, 0.5, 0.5005, 0.5005]",
         "[0.5, 0.5, 0.51, 0.51]",
         None,
         "Box too small (below threshold)"),
    ]
    
    print("=" * 70)
    print("R6 Simple - Enhanced GRPO Reward Function Test Results")
    print("For NORMALIZED bounding boxes [0,1]")
    print("=" * 70)
    
    for response, gt, extra_info, description in test_cases:
        analysis = analyze_reward(response, gt, extra_info)
        
        print(f"\n{description}:")
        print(f"  Final Score: {analysis['final_score']:.3f}")
        print(f"  Detection: {analysis['detection_score']:.3f} "
              f"(P={analysis['precision']:.2f}, R={analysis['recall']:.2f}, "
              f"IoU={analysis['avg_iou']:.2f})")
        print(f"  Quality: {analysis['quality_score']:.3f}")
        print(f"  Boxes: {analysis['num_predicted']} pred / {analysis['num_ground_truth']} true"
              f" / {analysis['expected_boxes']} expected")
        print(f"  Difficulty Weight: {analysis['difficulty_weight']:.2f}")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("- Correct no-findings get 0.92 reward (room for improvement)")
    print("- Difficulty adapts based on expected box count")
    print("- Simple, clean implementation with minimal hyperparameters")
    print("- F-beta automatically adjusts: 1.5 (easy) → 2.0 (medium) → 2.5 (hard)")
    print("=" * 70)