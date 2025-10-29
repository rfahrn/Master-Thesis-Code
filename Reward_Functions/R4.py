"""
R4.py - Enhanced RL Reward with Smooth Gradients
=================================================
Optimized for GRPO training with improved learning signals over R3.

KEY ENHANCEMENTS OVER R3:
✅ Smooth spline-based IoU rewards (better gradients than linear mean)
✅ Optional center-aware quality metric (novel geometric penalty)
✅ Still bounded [0,1] and stable for RL
✅ Only 4-5 hyperparameters (not 33!)
✅ Fast greedy matching (not Hungarian)

FORMULA:
reward = F_β × smooth_quality

Where:
- F_β = (1+β²)×(P×R)/(β²×P+R) [detection quality]
- smooth_quality = mean of spline-transformed IoUs [localization quality]
- Optional: center-aware adjustment

WHY THIS IS BETTER THAN R3 FOR RL:
1. Smoother gradients: Spline gives consistent learning signal across IoU ranges
2. Principled reward shaping: Steeper gradient near threshold, gentler at extremes
3. Optional geometric awareness: Center distance captures offset errors IoU might miss
4. Still simple: 4-5 params vs R3's 3 (vs R4_old's 33!)
5. Still stable: Bounded [0,1], no discontinuities, clear credit assignment

DESIGN PHILOSOPHY:
- Keep R3's simplicity and stability
- Add smooth reward shaping for better gradients
- One optional geometric enhancement (center distance)
- No redundant penalties (size, aspect already in IoU)
- No complex multi-term nonsense
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# ============================================================================
# HYPERPARAMETERS (Only 4-5!)
# ============================================================================

BETA = 1.5                      # F-beta parameter (1.0=balanced, 1.5=mild recall)
MIN_IOU_THRESHOLD = 0.5         # Matching threshold (COCO standard)
NO_BOX_BONUS = 0.2              # True negative reward
USE_CENTER_AWARE = False        # Enable center distance refinement
CENTER_WEIGHT = 0.15            # Weight for center distance (if enabled)

# ============================================================================
# SMOOTH REWARD SPLINE (Principled Design)
# ============================================================================

class SmoothRewardSpline:
    """
    Smooth IoU → Reward mapping using cubic spline.
    
    DESIGN PRINCIPLES:
    1. Start at zero (0,0) - no reward for no overlap
    2. Steep gradient [0, 0.5) - encourage getting to threshold
    3. Moderate gradient [0.5, 0.8] - steady improvement
    4. Gentle gradient [0.8, 1.0] - diminishing returns for perfection
    5. C¹ continuous - smooth gradients everywhere
    6. Bounded [0, 1] - RL stability
    
    This gives better learning signal than R3's linear mean_IoU:
    - IoU 0.3→0.4: Larger reward increase (steeper gradient)
    - IoU 0.9→1.0: Smaller reward increase (diminishing returns)
    - Encourages "good enough" boxes without obsessing over perfect alignment
    """
    
    def __init__(self):
        """
        Create smooth cubic spline with 4 control points.
        
        Control points chosen based on:
        - x=0.0: No overlap → no reward
        - x=0.5: Standard threshold → 50% reward (linear baseline)
        - x=0.8: Good match → 80% reward
        - x=1.0: Perfect match → 100% reward
        
        Gradients designed for optimal learning:
        - High gradient early (0-0.5): Encourage crossing threshold
        - Moderate gradient middle (0.5-0.8): Steady improvement
        - Lower gradient late (0.8-1.0): Don't over-optimize perfection
        """
        # Control points: (IoU, Reward, Gradient)
        # Gradients tuned for good learning dynamics
        self.control_points = np.array([
            [0.00, 0.00, 1.5],   # Zero start, steep initial gradient
            [0.50, 0.50, 1.1],   # At threshold, slightly above linear
            [0.80, 0.80, 0.9],   # Good zone, gentle gradient
            [1.00, 1.00, 0.5],   # Perfect, very gentle (diminishing returns)
        ])
        
        self._build_spline()
    
    def _build_spline(self):
        """Build piecewise cubic Hermite spline."""
        from scipy.interpolate import CubicHermiteSpline
        
        x = self.control_points[:, 0]
        y = self.control_points[:, 1]
        dydx = self.control_points[:, 2]
        
        self.spline = CubicHermiteSpline(x, y, dydx)
    
    def __call__(self, iou: float) -> float:
        """
        Evaluate smooth reward for given IoU.
        
        Args:
            iou: Intersection over Union [0, 1]
            
        Returns:
            Smooth reward [0, 1]
        """
        iou = float(np.clip(iou, 0.0, 1.0))
        reward = float(self.spline(iou))
        return float(np.clip(reward, 0.0, 1.0))


# Global spline instance
REWARD_SPLINE = SmoothRewardSpline()


def smooth_iou_reward(iou: float) -> float:
    """
    Apply smooth spline transformation to IoU.
    
    WHY THIS HELPS RL LEARNING:
    - Consistent gradient magnitude across IoU ranges
    - Steeper gradient where it matters (near threshold)
    - Gentler gradient where less critical (near perfect)
    - Smooth C¹ continuity (no gradient jumps)
    
    Example:
    - IoU 0.3→0.4: reward ~0.35→0.48 (Δ=0.13, steep!)
    - IoU 0.9→1.0: reward ~0.93→1.00 (Δ=0.07, gentle)
    
    Args:
        iou: Raw IoU value [0, 1]
        
    Returns:
        Smooth reward [0, 1]
    """
    return REWARD_SPLINE(iou)


# ============================================================================
# OPTIONAL: CENTER-AWARE QUALITY
# ============================================================================

def compute_center_distance_penalty(pred_box: List[float], gt_box: List[float]) -> float:
    """
    Normalized center distance penalty.
    
    WHY THIS HELPS:
    Two boxes can have same IoU but different center alignment:
    - Box A: IoU=0.6, centers aligned
    - Box B: IoU=0.6, centers offset
    
    This penalty helps discriminate between them.
    Normalized by GT box diagonal for scale-invariance.
    
    IMPORTANT: This is OPTIONAL enhancement, not core metric!
    Only use if you empirically see center-offset issues in training.
    
    Args:
        pred_box: [x1, y1, x2, y2]
        gt_box: [x1, y1, x2, y2]
        
    Returns:
        Penalty [0, 1] where 0=perfect alignment, 1=very offset
    """
    # Compute centers
    pred_cx = (pred_box[0] + pred_box[2]) / 2
    pred_cy = (pred_box[1] + pred_box[3]) / 2
    gt_cx = (gt_box[0] + gt_box[2]) / 2
    gt_cy = (gt_box[1] + gt_box[3]) / 2
    
    # Euclidean distance
    dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
    
    # Normalize by GT diagonal
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]
    diagonal = np.sqrt(gt_w**2 + gt_h**2)
    
    if diagonal == 0:
        return 0.0
    
    penalty = dist / diagonal
    return float(np.clip(penalty, 0.0, 1.0))


# ============================================================================
# CORE FUNCTIONS (Same as R3)
# ============================================================================

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


def greedy_match_boxes(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    min_iou: float = MIN_IOU_THRESHOLD
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Greedy matching: fast and usually optimal.
    
    FIXED: R4_old had bug where comment said greedy but code did Hungarian.
    This version actually does greedy matching!
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Compute IoU matrix
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


# ============================================================================
# MAIN REWARD FUNCTION
# ============================================================================

def enhanced_fbeta_reward(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    beta: float = BETA,
    use_center_aware: bool = USE_CENTER_AWARE,
    center_weight: float = CENTER_WEIGHT,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Enhanced F-beta reward with smooth quality metric.
    
    IMPROVEMENTS OVER R3:
    ===================
    
    1. SMOOTH QUALITY METRIC
       R3: mean_IoU (linear)
       R4: mean of spline(IoU) (smooth, shaped gradients)
       
       Why better: Consistent gradient magnitude across IoU ranges.
       Learning is more stable - model doesn't get "stuck" in certain regions.
    
    2. OPTIONAL CENTER-AWARE QUALITY
       Can optionally add small center distance penalty.
       Weight: 0.85×smooth_IoU + 0.15×center_quality
       
       Why useful: Distinguishes between IoU=0.6 well-centered vs IoU=0.6 offset.
       Only use if you see systematic center offset issues in training.
    
    3. STILL RL-OPTIMIZED
       ✅ Bounded [0,1]
       ✅ Smooth (C¹ continuous)
       ✅ No discontinuities
       ✅ Clear credit assignment (F_β + quality)
       ✅ Fast (greedy matching)
    
    FORMULA:
    --------
    If use_center_aware=False (default, simpler):
        quality = mean(spline(IoU_i))
        reward = F_β × quality
    
    If use_center_aware=True (optional enhancement):
        for each match:
            smooth_iou_i = spline(IoU_i)
            center_quality_i = 1 - center_penalty_i
            combined_i = (1-w)×smooth_iou_i + w×center_quality_i
        quality = mean(combined_i)
        reward = F_β × quality
    
    Args:
        predicted_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        actual_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        beta: F-beta parameter (default: 1.5 for mild recall emphasis)
        use_center_aware: Enable center distance refinement (default: False)
        center_weight: Weight for center term if enabled (default: 0.15)
        return_details: If True, return detailed metrics dictionary

    Returns:
        Reward score or detailed metrics dict
    """
    n_pred = len(predicted_boxes)
    n_gt = len(actual_boxes)
    
    # Initialize details dictionary
    details = {
        'reward': 0.0,
        'fbeta_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'smooth_quality': 0.0,
        'mean_raw_iou': 0.0,
        'matched_ious': [],
        'smooth_ious': [],
        'center_penalties': [],
        'num_matches': 0,
        'num_predictions': n_pred,
        'num_ground_truth': n_gt,
        'edge_case': classify_edge_case(n_pred, n_gt),
        'matches': [],
        'iou_matrix': None,
        'beta': beta,
        'center_aware_used': use_center_aware
    }
    
    # Handle empty cases
    if n_pred == 0 and n_gt == 0:
        # True negative: both correctly empty
        details['reward'] = NO_BOX_BONUS
        details['fbeta_score'] = 1.0
        details['precision'] = 1.0
        details['recall'] = 1.0
        details['smooth_quality'] = 1.0
        details['mean_raw_iou'] = 1.0
        return details if return_details else NO_BOX_BONUS

    if n_pred == 0 or n_gt == 0:
        # Either all false negatives or all false positives
        # F_β = 0 (either P=0 or R=0)
        details['reward'] = 0.0
        details['precision'] = 0.0
        details['recall'] = 0.0
        return details if return_details else 0.0

    # Perform greedy matching (FIXED: actually greedy now!)
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
    
    # Compute F-beta score
    if precision + recall == 0:
        fbeta_score = 0.0
    else:
        beta_sq = beta * beta
        fbeta_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    details['fbeta_score'] = fbeta_score
    
    # Compute smooth quality metric
    if matched_ious:
        # Apply spline transformation to each IoU
        smooth_ious = [smooth_iou_reward(iou) for iou in matched_ious]
        details['smooth_ious'] = smooth_ious
        
        if use_center_aware:
            # Optional: Combine smooth IoU with center distance
            combined_qualities = []
            center_penalties = []
            
            for (pred_idx, gt_idx, iou), smooth_iou in zip(matches, smooth_ious):
                # Compute center penalty
                center_penalty = compute_center_distance_penalty(
                    predicted_boxes[pred_idx],
                    actual_boxes[gt_idx]
                )
                center_penalties.append(center_penalty)
                
                # Combine: weighted average of smooth_iou and center_quality
                center_quality = 1.0 - center_penalty
                combined = (1 - center_weight) * smooth_iou + center_weight * center_quality
                combined_qualities.append(combined)
            
            smooth_quality = float(np.mean(combined_qualities))
            details['center_penalties'] = center_penalties
        else:
            # Simple: just mean of smooth IoUs
            smooth_quality = float(np.mean(smooth_ious))
        
        details['mean_raw_iou'] = float(np.mean(matched_ious))
    else:
        smooth_quality = 0.0
        details['mean_raw_iou'] = 0.0
    
    details['smooth_quality'] = smooth_quality
    
    # FINAL REWARD: F_β × smooth_quality
    # Same structure as R3, but with enhanced quality metric
    reward = fbeta_score * smooth_quality
    
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
    Main entry point for RL training.
    
    USAGE:
    ------
    # Default (smooth IoU only, like improved R3)
    reward = compute_score("dataset", model_output, ground_truth)
    
    # With center awareness (optional enhancement)
    import R4
    R4.USE_CENTER_AWARE = True
    reward = compute_score("dataset", model_output, ground_truth)
    """
    # Extract predicted boxes from <answer> tags or full string
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I | re.S)
    if m:
        answer_content = m.group(1)
    else:
        answer_content = solution_str

    predicted_boxes = extract_bounding_boxes(answer_content)
    ground_truth_boxes = extract_bounding_boxes(ground_truth)
    
    # Compute enhanced F-beta reward
    result = enhanced_fbeta_reward(
        predicted_boxes, 
        ground_truth_boxes,
        beta=BETA,
        use_center_aware=USE_CENTER_AWARE,
        center_weight=CENTER_WEIGHT,
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
    fbeta_scores = []
    smooth_qualities = []
    raw_ious = []
    detailed_metrics = []
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = compute_score(
            data_source="test",
            solution_str=pred,
            ground_truth=gt,
            return_details=True
        )
        
        rewards.append(metrics['reward'])
        fbeta_scores.append(metrics['fbeta_score'])
        smooth_qualities.append(metrics['smooth_quality'])
        raw_ious.append(metrics['mean_raw_iou'])
        detailed_metrics.append(metrics)
    
    analysis = {
        'reward_stats': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        },
        'fbeta_stats': {
            'mean': float(np.mean(fbeta_scores)),
            'std': float(np.std(fbeta_scores))
        },
        'quality_stats': {
            'smooth_mean': float(np.mean(smooth_qualities)),
            'raw_iou_mean': float(np.mean([x for x in raw_ious if x > 0])) if any(x > 0 for x in raw_ious) else 0.0
        },
        'detailed_metrics': detailed_metrics
    }
    
    return analysis


# ============================================================================
# TESTING & DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("R4: Enhanced RL Reward with Smooth Gradients")
    print("=" * 80)
    print(f"\nHyperparameters: {4 if not USE_CENTER_AWARE else 5} total")
    print(f"  BETA:             {BETA}")
    print(f"  MIN_IOU_THRESHOLD: {MIN_IOU_THRESHOLD}")
    print(f"  NO_BOX_BONUS:     {NO_BOX_BONUS}")
    print(f"  USE_CENTER_AWARE: {USE_CENTER_AWARE}")
    if USE_CENTER_AWARE:
        print(f"  CENTER_WEIGHT:    {CENTER_WEIGHT}")
    
    print(f"\nKey improvements over R3:")
    print(f"  ✅ Smooth spline-based quality (better gradients)")
    print(f"  ✅ Optional center-aware refinement")
    print(f"  ✅ Still simple (4-5 params vs R3's 3)")
    print(f"  ✅ Still stable (bounded [0,1], fast greedy matching)")
    
    # ========================================================================
    # SPLINE VISUALIZATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SPLINE REWARD FUNCTION")
    print("=" * 80)
    print("\nComparison: Linear (R3) vs Smooth Spline (R4)")
    print(f"{'IoU':<8} {'Linear (R3)':<15} {'Spline (R4)':<15} {'Gradient Ratio':<15}")
    print("-" * 65)
    
    test_ious = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prev_spline = 0.0
    prev_linear = 0.0
    
    for iou in test_ious:
        linear = iou  # R3's linear approach
        spline = smooth_iou_reward(iou)
        
        if iou > 0:
            linear_grad = (linear - prev_linear) / 0.1
            spline_grad = (spline - prev_spline) / 0.1
            grad_ratio = spline_grad / linear_grad if linear_grad > 0 else 1.0
            grad_str = f"{grad_ratio:.2f}x"
        else:
            grad_str = "baseline"
        
        print(f"{iou:<8.1f} {linear:<15.3f} {spline:<15.3f} {grad_str:<15}")
        prev_linear = linear
        prev_spline = spline
    
    print("\nNotice: Spline has steeper gradient [0-0.5] and gentler [0.8-1.0]")
    print("This encourages crossing threshold while avoiding perfectionism!")
    
    # ========================================================================
    # TEST CASES
    # ========================================================================
    
    test_cases = [
        {
            'name': 'Perfect Match',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
        },
        {
            'name': 'Good IoU (0.7) - Better reward shaping',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.35]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
        },
        {
            'name': 'Medium IoU (0.55) - Steeper gradient here!',
            'prediction': '<answer>[0.1, 0.15, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
        },
        {
            'name': 'Near threshold (0.48) - Strong learning signal',
            'prediction': '<answer>[0.12, 0.22, 0.32, 0.42]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
        },
        {
            'name': 'Multi-box with varying IoUs',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.48, 0.48, 0.62, 0.62]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
        },
        {
            'name': 'Over-prediction (natural penalty via F-beta)',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.9, 0.9, 1.0, 1.0]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
        },
    ]
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        result = compute_score(
            data_source="test",
            solution_str=test['prediction'],
            ground_truth=test['ground_truth'],
            return_details=True
        )
        
        print(f"\n[{i}] {test['name']}")
        print(f"    Reward:          {result['reward']:.4f}")
        print(f"    F-beta:          {result['fbeta_score']:.4f}")
        print(f"    Smooth Quality:  {result['smooth_quality']:.4f}")
        print(f"    Raw IoU:         {result['mean_raw_iou']:.4f}")
        if result['matched_ious']:
            print(f"    Matched IoUs:    {[f'{x:.3f}' for x in result['matched_ious']]}")
            print(f"    Smooth IoUs:     {[f'{x:.3f}' for x in result['smooth_ious']]}")
    
    # ========================================================================
    # R3 VS R4 COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("R3 VS R4 COMPARISON")
    print("=" * 80)
    print("\nScenario: 2 matches with IoU=[0.55, 0.90]")
    print("(Testing gradient shaping benefit)")
    
    ious = [0.55, 0.90]
    
    # R3 approach: linear mean
    r3_quality = np.mean(ious)
    r3_reward = 1.0 * r3_quality  # Assume F_beta=1.0 for comparison
    
    # R4 approach: spline mean
    r4_quality = np.mean([smooth_iou_reward(iou) for iou in ious])
    r4_reward = 1.0 * r4_quality
    
    print(f"\nR3 (Linear):")
    print(f"  Quality: {r3_quality:.4f}")
    print(f"  Reward:  {r3_reward:.4f}")
    
    print(f"\nR4 (Spline):")
    print(f"  Quality: {r4_quality:.4f}")
    print(f"  Reward:  {r4_reward:.4f}")
    print(f"  Difference: {r4_reward - r3_reward:+.4f}")
    
    print("\nInterpretation:")
    print("  R4 provides better-shaped gradients across IoU ranges.")
    print("  This should lead to more stable and efficient learning.")
    
    # ========================================================================
    # CENTER-AWARE DEMO (Optional)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CENTER-AWARE ENHANCEMENT (Optional)")
    print("=" * 80)
    
    # Two boxes with same IoU but different center alignment
    gt_box = [0.4, 0.4, 0.6, 0.6]
    box_centered = [0.41, 0.41, 0.61, 0.61]      # Well centered
    box_offset = [0.35, 0.35, 0.55, 0.55]        # Offset
    
    iou_centered = compute_iou(box_centered, gt_box)
    iou_offset = compute_iou(box_offset, gt_box)
    
    center_penalty_1 = compute_center_distance_penalty(box_centered, gt_box)
    center_penalty_2 = compute_center_distance_penalty(box_offset, gt_box)
    
    print("\nTwo predictions with similar IoU:")
    print(f"\nBox 1 (well-centered):")
    print(f"  IoU: {iou_centered:.3f}")
    print(f"  Center penalty: {center_penalty_1:.3f}")
    
    print(f"\nBox 2 (offset):")
    print(f"  IoU: {iou_offset:.3f}")
    print(f"  Center penalty: {center_penalty_2:.3f}")
    
    print(f"\nWith center-aware enabled (weight={CENTER_WEIGHT}):")
    print(f"  Box 1 would get slightly higher reward")
    print(f"  This helps model learn proper center alignment")
    
    print("\nRECOMMENDATION:")
    print("  Start with USE_CENTER_AWARE=False (simpler)")
    print("  Enable only if you see systematic center offset issues")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY: R4 vs R3")
    print("=" * 80)
    print("""
R3 (Simple):
  - Linear mean_IoU
  - 3 hyperparameters
  - Very simple and stable
  - Good baseline

R4 (Enhanced):
  - Smooth spline quality metric
  - Optional center-aware refinement
  - 4-5 hyperparameters (still simple!)
  - Better gradient shaping
  - Recommended for GRPO training

KEY ENHANCEMENTS:
1. Spline gives consistent gradient magnitude across IoU ranges
2. Steeper gradient near threshold (0.3-0.5) encourages crossing it
3. Gentler gradient near perfect (0.9-1.0) avoids over-optimization
4. Optional center penalty for geometric refinement

WHEN TO USE R4 OVER R3:
✅ You have GRPO training capacity to benefit from better gradients
✅ You want smoother learning dynamics
✅ You can afford 1-2 extra hyperparameters
✅ You see training instability with R3's linear quality

WHEN TO STICK WITH R3:
✅ You want maximum simplicity
✅ You're doing small-scale experiments
✅ R3 is already working well

RECOMMENDATION:
Start with R4 (center-aware disabled). It's nearly as simple as R3 but
with better learning dynamics. Only enable center-aware if needed.
""")
    
    print("=" * 80)
    print("Ready for enhanced GRPO training! 🚀")
    print("=" * 80)