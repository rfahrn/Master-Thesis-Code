"""
R2.py - F1-Weighted IoU Reward

"""
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# Hyperparameters
NO_BOX_BONUS = 0.2      # Reward for correct negative predictions (both empty)
MIN_IOU_THRESHOLD = 0.5  # Minimum IoU to consider a match (standard: 0.5)


def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """
    Extract bounding boxes [x1, y1, x2, y2] from answer string.
    
    Args:
        answer: String containing bounding boxes in format [x1,y1,x2,y2]
        
    Returns:
        List of bounding boxes as [x1, y1, x2, y2] coordinates
    """
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
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value in [0, 1]
    """
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


def greedy_match_boxes(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    min_iou: float = MIN_IOU_THRESHOLD
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Perform greedy matching between predicted and ground truth boxes.
    
    Strategy: Sort predictions by maximum IoU, then greedily match
    each prediction to the best available ground truth box.
    
    IMPORTANT: This function NEVER modifies box coordinates!
    It only determines which predictions correspond to which ground truths.
    
    Args:
        predicted_boxes: List of predicted bounding boxes [x1,y1,x2,y2]
        actual_boxes: List of ground truth bounding boxes [x1,y1,x2,y2]
        min_iou: Minimum IoU threshold to consider a match
        
    Returns:
        matches: List of (pred_idx, gt_idx, iou) tuples
        iou_matrix: Full IoU matrix (n_pred × n_gt)
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Compute IoU matrix (all pairwise comparisons)
    # Each IoU is computed from the ORIGINAL box coordinates
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(actual_boxes):
            ious[i, j] = compute_iou(pred, gt)
    
    # Greedy matching: sort by maximum IoU
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    max_ious = np.max(ious, axis=1) if n_gt > 0 else np.zeros(n_pred)
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
        if best_iou >= min_iou:
            matched_gt.add(best_j)
            matched_pred.add(i)
            matches.append((i, best_j, float(best_iou)))
    
    return matches, ious


def f1_iou_reward(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute F1-weighted IoU reward for grounding tasks.
    
    REWARD FORMULA:
    reward = F1-score × mean_IoU
    
    Where:
    - F1 = 2×P×R/(P+R), with P=precision, R=recall
    - Precision = num_matches / num_predictions
    - Recall = num_matches / num_ground_truths  
    - mean_IoU = average IoU of matched boxes
    
    SPECIAL CASES:
    - Both empty (true negative): reward = NO_BOX_BONUS (0.2)
    - Either empty: reward = 0.0 (F1 = 0)
    - No matches above threshold: reward = 0.0
    
    CHARACTERISTICS:
    ✅ Bounded [0, 1] (except true negative bonus)
    ✅ Balances precision-recall trade-offs
    ✅ Penalizes poor localization via IoU weighting
    ✅ Smooth, continuous gradients
    ✅ Handles multi-box scenarios correctly
    
    EXAMPLES:
    - Perfect: 10 boxes, 10 matches, IoU=1.0 → F1=1.0 × IoU=1.0 = 1.0
    - Good P/R, poor loc: 10/10, IoU=0.5 → F1=1.0 × IoU=0.5 = 0.5
    - Poor precision: 100 pred, 10 GT, 10 matches → F1=0.18 × IoU=x
    - Hallucination: 10 pred, 0 GT → F1=0.0 → reward=0.0
    
    Args:
        predicted_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        actual_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        return_details: If True, return detailed metrics dictionary

    Returns:
        F1×IoU reward score or detailed metrics dict
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Initialize details dictionary
    details = {
        'reward': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'mean_iou': 0.0,
        'matched_ious': [],
        'num_matches': 0,
        'num_predictions': n_pred,
        'num_ground_truth': n_gt,
        'edge_case': classify_edge_case(n_pred, n_gt),
        'matches': [],
        'iou_matrix': None
    }
    
    # Handle empty cases
    if n_pred == 0 and n_gt == 0:
        # True negative: both correctly empty
        details['reward'] = NO_BOX_BONUS
        details['f1_score'] = 1.0  # Perfect detection (of nothing)
        details['precision'] = 1.0
        details['recall'] = 1.0
        details['mean_iou'] = 1.0  # Conceptually perfect
        return details if return_details else NO_BOX_BONUS

    if n_pred == 0 or n_gt == 0:
        # Either all false negatives or all false positives
        # F1 = 0 (either P=0 or R=0)
        details['reward'] = 0.0
        details['precision'] = 0.0 if n_pred == 0 else 0.0
        details['recall'] = 0.0 if n_gt == 0 else 0.0
        return details if return_details else 0.0

    # Perform greedy matching
    matches, iou_matrix = greedy_match_boxes(predicted_boxes, actual_boxes)
    details['iou_matrix'] = iou_matrix
    details['matches'] = matches
    
    # Extract match statistics
    num_matches = len(matches)
    matched_ious = [iou for _, _, iou in matches]
    
    details['num_matches'] = num_matches
    details['matched_ious'] = matched_ious
    
    # Compute precision and recall
    precision = num_matches / n_pred if n_pred > 0 else 0.0
    recall = num_matches / n_gt if n_gt > 0 else 0.0
    
    details['precision'] = precision
    details['recall'] = recall
    
    # Compute F1-score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    details['f1_score'] = f1_score
    
    # Compute mean IoU of matches (localization quality)
    if matched_ious:
        mean_iou = float(np.mean(matched_ious))
    else:
        mean_iou = 0.0
    
    details['mean_iou'] = mean_iou
    
    # FINAL REWARD: F1 × mean_IoU
    # This naturally balances detection quality and localization quality
    reward = f1_score * mean_iou
    
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
    Compute F1-weighted IoU reward for grounding tasks.
    
    This is the main entry point used by the RL training loop.
    
    REWARD FORMULA:
    reward = F1-score × mean_IoU
    
    ADVANTAGES OVER PENALTY-BASED APPROACHES:
    ✅ Proper precision-recall balance (not arbitrary penalties)
    ✅ Bounded [0, 1] (stable for RL training)
    ✅ Scales correctly across different dataset sizes
    ✅ Natural localization quality penalty
    ✅ Interpretable: 0.7 means "70% of perfect detection+localization"
    
    EXAMPLES:
    - Perfect match: F1=1.0 × IoU=1.0 → reward = 1.0
    - Good detection, poor loc: F1=1.0 × IoU=0.5 → reward = 0.5
    - Poor precision (hallucinations): F1=0.2 × IoU=1.0 → reward = 0.2
    - Partial recall: F1=0.67 × IoU=0.8 → reward = 0.54
    - Complete miss: F1=0.0 → reward = 0.0

    Args:
        data_source: Name of the dataset
        solution_str: Model's output string (detokenized)
        ground_truth: Ground truth boxes "[x1,y1,x2,y2],[x1,y1,x2,y2],..."
        extra_info: Additional information (optional)
        return_details: If True, return detailed metrics dictionary

    Returns:
        F1×IoU reward score or detailed metrics dictionary
    """
    # Extract predicted boxes from <answer> tags or full string
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I | re.S)
    if m:
        answer_content = m.group(1)
    else:
        answer_content = solution_str

    predicted_boxes = extract_bounding_boxes(answer_content)
    ground_truth_boxes = extract_bounding_boxes(ground_truth)
    
    # Compute F1-weighted IoU reward
    result = f1_iou_reward(
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
    """
    Analyze reward distribution across a dataset.
    
    Provides comprehensive statistics including:
    - Overall reward statistics (mean, std, percentiles)
    - Per-edge-case breakdowns
    - F1, precision, recall distributions
    - IoU quality distributions
    """
    rewards = []
    f1_scores = []
    precisions = []
    recalls = []
    mean_ious = []
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
        f1_scores.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        mean_ious.append(metrics['mean_iou'])
        detailed_metrics.append(metrics)
        
        edge_case = metrics['edge_case']
        if edge_case not in edge_cases:
            edge_cases[edge_case] = {
                'rewards': [],
                'f1_scores': [],
                'precisions': [],
                'recalls': [],
                'mean_ious': []
            }
        edge_cases[edge_case]['rewards'].append(metrics['reward'])
        edge_cases[edge_case]['f1_scores'].append(metrics['f1_score'])
        edge_cases[edge_case]['precisions'].append(metrics['precision'])
        edge_cases[edge_case]['recalls'].append(metrics['recall'])
        edge_cases[edge_case]['mean_ious'].append(metrics['mean_iou'])
    
    rewards_array = np.array(rewards)
    
    analysis = {
        'reward_stats': {
            'mean': float(np.mean(rewards_array)),
            'std': float(np.std(rewards_array)),
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'median': float(np.median(rewards_array)),
            'percentiles': {
                '25': float(np.percentile(rewards_array, 25)),
                '50': float(np.percentile(rewards_array, 50)),
                '75': float(np.percentile(rewards_array, 75)),
                '90': float(np.percentile(rewards_array, 90)),
                '95': float(np.percentile(rewards_array, 95))
            }
        },
        'f1_stats': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores))
        },
        'precision_stats': {
            'mean': float(np.mean(precisions)),
            'std': float(np.std(precisions))
        },
        'recall_stats': {
            'mean': float(np.mean(recalls)),
            'std': float(np.std(recalls))
        },
        'iou_stats': {
            'mean': float(np.mean([iou for iou in mean_ious if iou > 0])) if any(iou > 0 for iou in mean_ious) else 0.0,
            'std': float(np.std([iou for iou in mean_ious if iou > 0])) if any(iou > 0 for iou in mean_ious) else 0.0
        },
        'edge_case_distribution': {
            case: {
                'count': len(case_data['rewards']),
                'mean_reward': float(np.mean(case_data['rewards'])) if case_data['rewards'] else 0.0,
                'mean_f1': float(np.mean(case_data['f1_scores'])) if case_data['f1_scores'] else 0.0,
                'mean_precision': float(np.mean(case_data['precisions'])) if case_data['precisions'] else 0.0,
                'mean_recall': float(np.mean(case_data['recalls'])) if case_data['recalls'] else 0.0,
                'mean_iou': float(np.mean([iou for iou in case_data['mean_ious'] if iou > 0])) if any(iou > 0 for iou in case_data['mean_ious']) else 0.0
            }
            for case, case_data in edge_cases.items()
        },
        'total_samples': len(rewards_array),
        'detailed_metrics': detailed_metrics
    }
    
    return analysis


# ============================================================================
# TESTING & COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("R2: F1-Weighted IoU Reward Function")
    print("=" * 70)
    print("Proper precision-recall balance with localization quality weighting")
    print("=" * 70)
    
    test_cases = [
        {
            'name': 'Perfect Match (1 box)',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
            'expected': 'F1=1.0 × IoU=1.0 = 1.0'
        },
        {
            'name': 'Good IoU (0.7)',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.35]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
            'expected': 'F1=1.0 × IoU=0.7 ≈ 0.7'
        },
        {
            'name': 'Medium IoU (0.55)',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
            'expected': 'F1=1.0 × IoU=0.55 ≈ 0.55'
        },
        {
            'name': 'Below threshold (IoU=0.49)',
            'prediction': '<answer>[0.15, 0.25, 0.35, 0.45]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
            'expected': 'No match (IoU<0.5) → F1=0.0 → reward=0.0'
        },
        {
            'name': 'True Negative',
            'prediction': '<answer></answer>',
            'ground_truth': '',
            'expected': 'Both empty → reward = 0.2 (bonus)'
        },
        {
            'name': 'Hallucination (FP)',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '',
            'expected': 'Precision=0, Recall=undef → F1=0.0 → reward=0.0'
        },
        {
            'name': 'Missed Detection (FN)',
            'prediction': '<answer></answer>',
            'ground_truth': '[0.5, 0.5, 0.7, 0.7]',
            'expected': 'Precision=undef, Recall=0 → F1=0.0 → reward=0.0'
        },
        {
            'name': 'Multi-box Perfect',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
            'expected': 'F1=1.0 × IoU=1.0 = 1.0'
        },
        {
            'name': 'Multi-box Partial (missed 1 of 2)',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
            'expected': 'P=1.0, R=0.5 → F1=0.67 × IoU=1.0 ≈ 0.67'
        },
        {
            'name': 'Multi-box with Hallucination',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 1.0, 1.0]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2]',
            'expected': 'P=0.5, R=1.0 → F1=0.67 × IoU=1.0 ≈ 0.67'
        },
        {
            'name': '🚨 CRITICAL: Many hallucinations (10 GT, 100 pred, 10 match)',
            'prediction': '<answer>' + ', '.join([f'[{i*0.1}, {i*0.1}, {i*0.1+0.1}, {i*0.1+0.1}]' for i in range(100)]) + '</answer>',
            'ground_truth': ', '.join([f'[{i*0.1}, {i*0.1}, {i*0.1+0.1}, {i*0.1+0.1}]' for i in range(10)]),
            'expected': 'P=0.1, R=1.0 → F1=0.18 × IoU=1.0 ≈ 0.18 (LOW!)'
        },
    ]
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        result = compute_score(
            data_source="test",
            solution_str=test['prediction'],
            ground_truth=test['ground_truth'],
            return_details=True
        )
        
        print(f"\n[{i}] {test['name']}")
        print(f"    Expected: {test['expected']}")
        print(f"    ─" * 35)
        print(f"    Reward:    {result['reward']:.4f}")
        print(f"    F1-score:  {result['f1_score']:.4f}")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall:    {result['recall']:.4f}")
        print(f"    Mean IoU:  {result['mean_iou']:.4f}")
        print(f"    Matches:   {result['num_matches']}/{result['num_predictions']} pred, {result['num_matches']}/{result['num_ground_truth']} GT")
        if result['matched_ious']:
            print(f"    IoUs:      {[f'{iou:.3f}' for iou in result['matched_ious']]}")
    
    print("\n" + "=" * 70)
    print("COMPARISON: F1×IoU vs Penalty-Based Approach")
    print("=" * 70)
    print("\nScenario: 10 GT boxes, 100 predictions, 10 perfect matches")
    print("This tests how well each approach handles excessive hallucinations.")
    print()
    print("Penalty-Based (old):")
    print("  reward = (10×1.0 - 0.1×90 - 0.1×0) / 10")
    print("  reward = (10 - 9) / 10 = 0.1")
    print("  ❌ Doesn't reflect the 9% precision problem!")
    print()
    print("F1×IoU (new):")
    print("  P = 10/100 = 0.1, R = 10/10 = 1.0")
    print("  F1 = 2×0.1×1.0/(0.1+1.0) = 0.182")
    print("  reward = 0.182 × 1.0 = 0.182")
    print("  ✅ Properly reflects poor precision via F1-score!")
    print()
    print("=" * 70)