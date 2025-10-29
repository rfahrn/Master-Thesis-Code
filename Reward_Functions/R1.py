"""
R1.py - Enhanced
================
Simple grounding reward function (r1) Based on radvlm evaluation metrics (AP@0.5).

ENHANCEMENTS:
- Sorted greedy matching (prioritizes best IoU matches first)
- Edge case classification and analysis
- Detailed metrics reporting
"""
import re
from typing import List, Dict, Any
import numpy as np

# Hyperparameter: Bonus reward for correctly predicting "no box" when GT has no box
NO_BOX_BONUS = 0.2  # Small reward for correct negative predictions


def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """Extract bounding boxes [x1, y1, x2, y2] from answer string.

    Supports various formats and handles malformed boxes gracefully.
    """
    # Pattern to match [x1, y1, x2, y2] with floats (including negatives and scientific notation)
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"

    boxes: List[List[float]] = []
    for m in re.finditer(pattern, answer):
        try:
            b = [float(m.group(1)), float(m.group(2)),
                 float(m.group(3)), float(m.group(4))]
        except Exception:
            # Skip unparsable numbers
            continue
        # Drop non-finite coordinates
        if not all(np.isfinite(b)):
            continue
        boxes.append(b)
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    denom = box1_area + box2_area - inter_area
    return inter_area / denom if denom > 0 else 0.0


def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision given precision and recall arrays."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def classify_edge_case(n_pred: int, n_gt: int) -> str:
    """Classify the scenario for analysis."""
    if n_pred == 0 and n_gt == 0:
        return "true_negative"
    elif n_pred > 0 and n_gt == 0:
        return "hallucination"
    elif n_pred == 0 and n_gt > 0:
        return "missed_detection"
    elif n_pred == 1 and n_gt == 1:
        return "one_to_one"
    elif n_pred > 1 and n_gt == 1:
        return "many_to_one"
    elif n_pred == 1 and n_gt > 1:
        return "one_to_many"
    else:
        return "many_to_many"


def average_precision_at_iou(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute Average Precision at a given IoU threshold.
    This matches the evaluation metric used in radvlm.

    ENHANCED: Uses sorted greedy matching (best IoU matches first).

    Args:
        predicted_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        actual_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        iou_threshold: IoU threshold for considering a match (default: 0.5)
        return_details: If True, return detailed metrics dictionary

    Returns:
        Average Precision score (0.0 to 1.0) or detailed metrics dict
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Initialize details dictionary
    details = {
        'ap': 0.0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'precision': 0.0,
        'recall': 0.0,
        'edge_case': classify_edge_case(n_pred, n_gt),
        'matches': [],  # List of (pred_idx, gt_idx, iou)
        'iou_matrix': None,
        'num_predictions': n_pred,
        'num_ground_truth': n_gt
    }
    
    # Handle empty cases
    if not predicted_boxes and not actual_boxes:
        # Both empty: correct "no box" prediction - return NO_BOX_BONUS
        details['ap'] = NO_BOX_BONUS
        details['precision'] = 1.0
        details['recall'] = 1.0
        return details if return_details else NO_BOX_BONUS

    if not predicted_boxes:
        # No predictions but GT exists: all false negatives
        details['fn'] = n_gt
        return details if return_details else 0.0
    
    if not actual_boxes:
        # Predictions but no GT: all false positives (hallucination)
        details['fp'] = n_pred
        return details if return_details else 0.0

    # Compute IoU matrix between all predicted and ground truth boxes
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(actual_boxes):
            ious[i, j] = compute_iou(pred, gt)
    
    details['iou_matrix'] = ious

    # ENHANCED: Sorted greedy matching (prioritize best matches first)
    matched_gt = set()
    true_positives = np.zeros(n_pred)
    false_positives = np.zeros(n_pred)
    
    # Sort predictions by maximum IoU (process best matches first)
    max_ious = np.max(ious, axis=1)
    sorted_indices = np.argsort(-max_ious)  # Descending order
    
    for idx in sorted_indices:
        i = int(idx)
        
        # Check if this prediction's best IoU is below threshold
        if max_ious[i] < iou_threshold:
            false_positives[i] = 1
            continue
        
        # Find best unmatched GT box
        available_gt = [j for j in range(n_gt) if j not in matched_gt]
        if not available_gt:
            false_positives[i] = 1
            continue
        
        # Select GT with highest IoU among available
        best_j = max(available_gt, key=lambda j: ious[i, j])
        
        if ious[i, best_j] >= iou_threshold:
            true_positives[i] = 1
            matched_gt.add(best_j)
            details['matches'].append((i, best_j, float(ious[i, best_j])))
        else:
            false_positives[i] = 1

    # Compute metrics
    details['tp'] = int(np.sum(true_positives))
    details['fp'] = int(np.sum(false_positives))
    details['fn'] = n_gt - len(matched_gt)
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Final metrics
    if len(precision) > 0:
        details['precision'] = float(precision[-1])
        details['recall'] = float(recall[-1])
    
    # Compute Average Precision
    ap = compute_average_precision(recall, precision)
    details['ap'] = ap
    
    return details if return_details else ap


def compute_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info=None,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute reward score for grounding tasks using AP@0.5.

    This is the main reward function called by veRL's RewardManager.
    Expected model output format (direct answer, no reasoning):
        <answer>[x1,y1,x2,y2],[x1,y1,x2,y2]</answer>

    REWARD DISTRIBUTION:
    ┌─────────────────────────────────────┬──────────────────────────────┐
    │ Scenario                            │ Reward                       │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ No boxes predicted, no GT boxes     │ NO_BOX_BONUS (0.2)          │
    │ (Correct negative prediction)       │ Small reward for correctness │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ Boxes predicted, but no GT boxes    │ 0.0 (False positives)       │
    │ (Model hallucinates boxes)          │                              │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ No boxes predicted, GT has boxes    │ 0.0 (Missed detections)     │
    │ (Model misses findings)             │                              │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ Boxes predicted, GT has boxes       │ AP@0.5 (0.0 to 1.0)         │
    │ - Perfect localization              │ → 1.0                        │
    │ - Good localization (IoU ≥ 0.5)     │ → 0.5 to 1.0                │
    │ - Poor localization (IoU < 0.5)     │ → 0.0 to 0.5                │
    │ - Completely wrong                  │ → 0.0                        │
    └─────────────────────────────────────┴──────────────────────────────┘

    Note: NO_BOX_BONUS (0.2) is intentionally lower than typical AP scores
    to encourage accurate localization while still rewarding correct negatives.

    Args:
        data_source: Name of the dataset (used to identify the task)
        solution_str: Model's output string (detokenized)
        ground_truth: Ground truth bounding boxes as string "[x1,y1,x2,y2],[x1,y1,x2,y2],..."
        extra_info: Additional information (optional, not used currently)
        return_details: If True, return detailed metrics dictionary

    Returns:
        Reward score between 0.0 and 1.0 (AP@0.5, or NO_BOX_BONUS for correct negatives)
        Or detailed metrics dictionary if return_details=True
    """
    # Extract predicted boxes from model output
    # Look for content within <answer>...</answer> tags
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I | re.S)
    if m:
        answer_content = m.group(1)
    else:
        # Fallback: if no tags, use the entire solution string
        answer_content = solution_str

    predicted_boxes = extract_bounding_boxes(answer_content)

    # Extract ground truth boxes (supports multiple boxes)
    ground_truth_boxes = extract_bounding_boxes(ground_truth)
    
    # Compute reward: AP@0.5 or NO_BOX_BONUS for correct negatives
    # Handles multi-box scenarios correctly with sorted greedy matching
    result = average_precision_at_iou(
        predicted_boxes, 
        ground_truth_boxes, 
        iou_threshold=0.5,
        return_details=return_details
    )
    
    if return_details:
        result['predicted_boxes'] = predicted_boxes
        result['ground_truth_boxes'] = ground_truth_boxes
        return result
    
    return result


def analyze_reward_distribution(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, Any]:
    """
    Analyze reward distribution across a dataset.
    
    Returns statistics about reward distribution and edge cases.
    """
    rewards = []
    edge_cases = {}
    detailed_metrics = []
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = compute_score(
            data_source="test",
            solution_str=pred,
            ground_truth=gt,
            return_details=True
        )
        rewards.append(metrics['ap'])
        detailed_metrics.append(metrics)
        
        edge_case = metrics['edge_case']
        if edge_case not in edge_cases:
            edge_cases[edge_case] = []
        edge_cases[edge_case].append(metrics['ap'])
    
    # Compute statistics
    rewards_array = np.array(rewards)
    
    analysis = {
        'mean_reward': float(np.mean(rewards_array)),
        'std_reward': float(np.std(rewards_array)),
        'min_reward': float(np.min(rewards_array)),
        'max_reward': float(np.max(rewards_array)),
        'median_reward': float(np.median(rewards_array)),
        'percentiles': {
            '25': float(np.percentile(rewards_array, 25)),
            '50': float(np.percentile(rewards_array, 50)),
            '75': float(np.percentile(rewards_array, 75)),
            '90': float(np.percentile(rewards_array, 90)),
            '95': float(np.percentile(rewards_array, 95))
        },
        'edge_case_distribution': {
            case: {
                'count': len(case_rewards),
                'mean_reward': float(np.mean(case_rewards)) if case_rewards else 0.0,
                'std_reward': float(np.std(case_rewards)) if len(case_rewards) > 1 else 0.0
            }
            for case, case_rewards in edge_cases.items()
        },
        'total_samples': len(rewards_array),
        'detailed_metrics': detailed_metrics
    }
    
    return analysis


# ============================================================================
# EDGE CASE TESTING & ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("R1 Grounding Reward Function - Enhanced Version (AP@0.5)")
    print("=" * 60)
    
    # Test cases demonstrating various scenarios
    test_cases = [
        {
            'name': 'Perfect Match',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'True Negative',
            'prediction': '<answer></answer>',
            'ground_truth': ''
        },
        {
            'name': 'Hallucination',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': ''
        },
        {
            'name': 'Missed Detection',
            'prediction': '<answer></answer>',
            'ground_truth': '[0.5, 0.5, 0.7, 0.7]'
        },
        {
            'name': 'Partial Overlap (IoU ~0.39)',
            'prediction': '<answer>[0.15, 0.25, 0.35, 0.45]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Good Overlap (IoU ~0.55)',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.35]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Multiple Boxes - Perfect',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]'
        },
        {
            'name': 'Multiple Boxes - Partial Match',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]'
        },
        {
            'name': 'Many-to-One (Multiple preds, one GT)',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'One-to-Many (One pred, multiple GT)',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.7]'
        },
        {
            'name': 'Multiple findings',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.6, 0.6]'
        }
    ]
    
    for test in test_cases:
        result = compute_score(
            data_source="test",
            solution_str=test['prediction'],
            ground_truth=test['ground_truth'],
            return_details=True
        )
        print(f"\n{test['name']}:")
        print(f"  Reward (AP): {result['ap']:.3f}")
        print(f"  Edge Case: {result['edge_case']}")
        print(f"  TP/FP/FN: {result['tp']}/{result['fp']}/{result['fn']}")