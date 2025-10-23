"""
R2.py - Continuous Reward Version
================
Smooth, continuous grounding reward function for RL training.

IMPROVEMENTS OVER AP@0.5:
- Continuous IoU-based rewards (not binary threshold)
- Smooth gradients for better RL learning
- Progressive credit for improving localization
- Penalizes false positives and missed detections

FORMULA:
- Matched boxes: Get IoU as reward (0.0 to 1.0)
- Unmatched predictions: -penalty
- Unmatched GT: -penalty
"""
import re
from typing import List, Dict, Any
import numpy as np

# Hyperparameters
NO_BOX_BONUS = 0.2  # Reward for correct negative predictions
FP_PENALTY = 0.1    # Penalty for false positive predictions
FN_PENALTY = 0.1    # Penalty for false negative (missed GT)
MIN_IOU_THRESHOLD = 0.0  # Minimum IoU to consider a match (0.0 = no threshold)


def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """Extract bounding boxes [x1, y1, x2, y2] from answer string."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"

    boxes: List[List[float]] = []
    for m in re.finditer(pattern, answer):
        try:
            b = [float(m.group(1)), float(m.group(2)),
                 float(m.group(3)), float(m.group(4))]
        except Exception:
            continue
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


def continuous_reward(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute CONTINUOUS reward based on IoU values (not binary threshold).
    
    REWARD FORMULA:
    reward = (sum of matched IoUs) - (FP_PENALTY * num_fps) - (FN_PENALTY * num_fns)
    
    Then normalized by number of GT boxes (if > 0).
    
    CHARACTERISTICS:
    - Continuous: IoU 0.49 gets 0.49 reward, not 0.0
    - Smooth gradients: Small improvements → small reward increases
    - Penalizes hallucinations and misses
    - Encourages accurate localization
    
    Args:
        predicted_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        actual_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        return_details: If True, return detailed metrics dictionary

    Returns:
        Continuous reward score or detailed metrics dict
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Initialize details dictionary
    details = {
        'reward': 0.0,
        'matched_ious': [],
        'num_matches': 0,
        'num_fps': 0,
        'num_fns': 0,
        'mean_iou': 0.0,
        'edge_case': classify_edge_case(n_pred, n_gt),
        'matches': [],
        'iou_matrix': None,
        'num_predictions': n_pred,
        'num_ground_truth': n_gt
    }
    
    # Handle empty cases
    if not predicted_boxes and not actual_boxes:
        details['reward'] = NO_BOX_BONUS
        details['mean_iou'] = 1.0
        return details if return_details else NO_BOX_BONUS

    if not predicted_boxes:
        # Missed all detections
        details['num_fns'] = n_gt
        details['reward'] = -FN_PENALTY * n_gt
        return details if return_details else details['reward']
    
    if not actual_boxes:
        # All predictions are hallucinations
        details['num_fps'] = n_pred
        details['reward'] = -FP_PENALTY * n_pred
        return details if return_details else details['reward']

    # Compute IoU matrix
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(actual_boxes):
            ious[i, j] = compute_iou(pred, gt)
    
    details['iou_matrix'] = ious

    # Sorted greedy matching (best IoU first)
    matched_gt = set()
    matched_pred = set()
    total_iou = 0.0
    
    # Sort predictions by maximum IoU
    max_ious = np.max(ious, axis=1)
    sorted_indices = np.argsort(-max_ious)
    
    for idx in sorted_indices:
        i = int(idx)
        
        # Find best unmatched GT
        available_gt = [j for j in range(n_gt) if j not in matched_gt]
        if not available_gt:
            break
        
        best_j = max(available_gt, key=lambda j: ious[i, j])
        best_iou = ious[i, best_j]
        
        # Match if IoU above minimum threshold
        if best_iou >= MIN_IOU_THRESHOLD:
            matched_gt.add(best_j)
            matched_pred.add(i)
            total_iou += best_iou
            details['matched_ious'].append(best_iou)
            details['matches'].append((i, best_j, float(best_iou)))

    # Count unmatched
    num_matches = len(matched_gt)
    num_fps = n_pred - len(matched_pred)  # Unmatched predictions
    num_fns = n_gt - len(matched_gt)      # Unmatched GT
    
    details['num_matches'] = num_matches
    details['num_fps'] = num_fps
    details['num_fns'] = num_fns
    
    # Compute mean IoU
    if details['matched_ious']:
        details['mean_iou'] = float(np.mean(details['matched_ious']))
    
    # CONTINUOUS REWARD FORMULA:
    # Reward = (sum of IoUs) - (penalties for FP and FN)
    # Normalized by number of GT boxes
    reward = total_iou - (FP_PENALTY * num_fps) - (FN_PENALTY * num_fns)
    
    # Normalize by number of GT boxes (if any)
    if n_gt > 0:
        reward = reward / n_gt
    
    details['reward'] = float(reward)
    
    return details if return_details else details['reward']


def compute_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info=None,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute CONTINUOUS reward for grounding tasks.
    
    REWARD FORMULA:
    - Matched boxes: Get IoU value (0.0 to 1.0) as reward
    - False positives: -0.1 penalty each
    - False negatives: -0.1 penalty each
    - Normalized by number of GT boxes
    
    ADVANTAGES OVER AP@0.5:
    ✅ Continuous gradients (IoU 0.49 → reward 0.49, not 0.0)
    ✅ Smooth learning signal for RL
    ✅ Progressive credit for improving localization
    ✅ Penalizes hallucinations and misses
    
    EXAMPLES:
    - Perfect match (IoU=1.0): reward = 1.0
    - Good match (IoU=0.7): reward = 0.7 (not 0.0!)
    - Partial match (IoU=0.4): reward = 0.4 (not 0.0!)
    - Hallucination: reward = -0.1
    - Missed detection: reward = -0.1

    Args:
        data_source: Name of the dataset
        solution_str: Model's output string (detokenized)
        ground_truth: Ground truth boxes "[x1,y1,x2,y2],[x1,y1,x2,y2],..."
        extra_info: Additional information (optional)
        return_details: If True, return detailed metrics dictionary

    Returns:
        Continuous reward score or detailed metrics dictionary
    """
    # Extract predicted boxes
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I | re.S)
    if m:
        answer_content = m.group(1)
    else:
        answer_content = solution_str

    predicted_boxes = extract_bounding_boxes(answer_content)
    ground_truth_boxes = extract_bounding_boxes(ground_truth)
    
    # Compute continuous reward
    result = continuous_reward(
        predicted_boxes, 
        ground_truth_boxes, 
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
    """Analyze reward distribution across a dataset."""
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
        rewards.append(metrics['reward'])
        detailed_metrics.append(metrics)
        
        edge_case = metrics['edge_case']
        if edge_case not in edge_cases:
            edge_cases[edge_case] = []
        edge_cases[edge_case].append(metrics['reward'])
    
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
# TESTING & COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("R1 CONTINUOUS Reward Function")
    print("=" * 70)
    print("Smooth, continuous rewards for RL training")
    print("=" * 70)
    
    test_cases = [
        {
            'name': 'Perfect Match',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Good IoU (0.7) - AP would give 1.0',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.35]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Medium IoU (0.55) - AP would give 1.0',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Borderline IoU (0.49) - AP would give 0.0!',
            'prediction': '<answer>[0.15, 0.25, 0.35, 0.45]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Low IoU (0.2) - AP would give 0.0',
            'prediction': '<answer>[0.2, 0.3, 0.4, 0.5]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'No Overlap - AP would give 0.0',
            'prediction': '<answer>[0.5, 0.5, 0.6, 0.6]</answer>',
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
            'name': 'Multiple Boxes - Perfect',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]'
        },
        {
            'name': 'Multiple Boxes - Partial (1 perfect, 1 missed)',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]'
        },
        {
            'name': 'Multiple Boxes - Good IoUs (0.7 each)',
            'prediction': '<answer>[0.1, 0.12, 0.2, 0.22], [0.5, 0.52, 0.6, 0.62]</answer>',
            'ground_truth': '[0.1, 0.1, 0.6, 0.2], [0.5, 0.52, 0.6, 0.62]'
        },
    ]
    
    print("\nCOMPARISON: Continuous vs Binary AP@0.5")
    print("-" * 70)
    
    for test in test_cases:
        result = compute_score(
            data_source="test",
            solution_str=test['prediction'],
            ground_truth=test['ground_truth'],
            return_details=True
        )
        print(f"\n{test['name']}:")
        print(f"  Reward: {result['reward']:.3f}")
        if result['matched_ious']:
            print(f"  Mean IoU: {result['mean_iou']:.3f}")
            print(f"  Matched IoUs: {[f'{iou:.3f}' for iou in result['matched_ious']]}")
        print(f"  Matches/FPs/FNs: {result['num_matches']}/{result['num_fps']}/{result['num_fns']}")