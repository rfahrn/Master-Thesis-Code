"""
Edge Case Test Suite for GPRO Reward Functions
================================================
Comprehensive test cases for evaluating reward function behavior across diverse scenarios.

This module defines edge cases organized by categories:
1. Basic Cases (true negative, perfect match, etc.)
2. Localization Quality (varying IoU levels)
3. Cardinality Mismatches (hallucinations, missed detections)
4. Multi-Box Scenarios (many-to-many, one-to-many, etc.)
5. Geometric Edge Cases (tiny boxes, large boxes, etc.)
6. Extreme Cases (overlapping, nested, etc.)

For use in Master Thesis: Comparative analysis of RL reward functions for radiology grounding.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class EdgeCaseTestSuite:
    """Comprehensive edge case test suite for reward function evaluation."""

    def __init__(self):
        """Initialize the test suite with all edge cases."""
        self.test_cases = self._build_test_cases()
        self.categories = self._categorize_tests()

    def _build_test_cases(self) -> List[Dict[str, Any]]:
        """Build comprehensive test cases."""
        return [
            # ================================================================
            # CATEGORY 1: BASIC CASES
            # ================================================================
            {
                'id': 'basic_001',
                'category': 'basic',
                'name': 'True Negative (Correct Empty Prediction)',
                'description': 'No findings predicted, no findings in ground truth',
                'prediction': '<answer></answer>',
                'ground_truth': '',
                'expected_behavior': 'Should reward correct negative prediction',
                'clinical_scenario': 'Normal chest X-ray, no abnormalities',
                'edge_case_type': 'true_negative'
            },
            {
                'id': 'basic_002',
                'category': 'basic',
                'name': 'Perfect Match (IoU = 1.0)',
                'description': 'Predicted box exactly matches ground truth',
                'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Maximum reward (1.0)',
                'clinical_scenario': 'Perfect localization of pneumonia',
                'edge_case_type': 'one_to_one',
                'expected_iou': 1.0
            },
            {
                'id': 'basic_003',
                'category': 'basic',
                'name': 'Hallucination (False Positive)',
                'description': 'Model predicts box where none exists',
                'prediction': '<answer>[0.5, 0.5, 0.7, 0.7]</answer>',
                'ground_truth': '',
                'expected_behavior': 'Zero or very low reward (penalty)',
                'clinical_scenario': 'Model hallucinates finding on normal scan',
                'edge_case_type': 'hallucination'
            },
            {
                'id': 'basic_004',
                'category': 'basic',
                'name': 'Missed Detection (False Negative)',
                'description': 'Model fails to predict existing finding',
                'prediction': '<answer></answer>',
                'ground_truth': '[0.2, 0.3, 0.6, 0.7]',
                'expected_behavior': 'Zero reward (missed critical finding)',
                'clinical_scenario': 'Missed tumor detection',
                'edge_case_type': 'missed_detection'
            },

            # ================================================================
            # CATEGORY 2: LOCALIZATION QUALITY (Varying IoU)
            # ================================================================
            {
                'id': 'iou_001',
                'category': 'localization',
                'name': 'Poor Localization (IoU ≈ 0.25)',
                'description': 'Box has some overlap but poor localization',
                'prediction': '<answer>[0.2, 0.3, 0.4, 0.5]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Low reward, below threshold',
                'clinical_scenario': 'Rough approximation of lesion location',
                'edge_case_type': 'one_to_one',
                'expected_iou': 0.25
            },
            {
                'id': 'iou_002',
                'category': 'localization',
                'name': 'Near-Threshold Localization (IoU ≈ 0.48)',
                'description': 'Just below matching threshold',
                'prediction': '<answer>[0.11, 0.21, 0.31, 0.41]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Below threshold, no match',
                'clinical_scenario': 'Almost correct lesion boundary',
                'edge_case_type': 'one_to_one',
                'expected_iou': 0.48
            },
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
            },
            {
                'id': 'iou_004',
                'category': 'localization',
                'name': 'Good Localization (IoU ≈ 0.70)',
                'description': 'Good overlap with ground truth',
                'prediction': '<answer>[0.105, 0.21, 0.295, 0.39]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'High reward',
                'clinical_scenario': 'Good lesion boundary estimation',
                'edge_case_type': 'one_to_one',
                'expected_iou': 0.70
            },
            {
                'id': 'iou_005',
                'category': 'localization',
                'name': 'Excellent Localization (IoU ≈ 0.90)',
                'description': 'Very close to perfect match',
                'prediction': '<answer>[0.101, 0.201, 0.299, 0.399]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Very high reward, near maximum',
                'clinical_scenario': 'Nearly perfect lesion segmentation',
                'edge_case_type': 'one_to_one',
                'expected_iou': 0.90
            },

            # ================================================================
            # CATEGORY 3: MULTI-BOX SCENARIOS
            # ================================================================
            {
                'id': 'multi_001',
                'category': 'multi_box',
                'name': 'Two Perfect Matches',
                'description': 'Two boxes, both perfectly matched',
                'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]</answer>',
                'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
                'expected_behavior': 'Maximum reward',
                'clinical_scenario': 'Multiple lesions, all correctly localized',
                'edge_case_type': 'many_to_many'
            },
            {
                'id': 'multi_002',
                'category': 'multi_box',
                'name': 'Partial Multi-Box Match',
                'description': 'Two predictions, one matches, one does not',
                'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.9, 0.9]</answer>',
                'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]',
                'expected_behavior': 'Partial reward (50% precision, 50% recall)',
                'clinical_scenario': 'Detected one lesion, missed another, hallucinated third',
                'edge_case_type': 'many_to_many'
            },
            {
                'id': 'multi_003',
                'category': 'multi_box',
                'name': 'Many-to-One (Multiple Predictions, One GT)',
                'description': 'Multiple predictions for single finding',
                'prediction': '<answer>[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45], [0.12, 0.22, 0.32, 0.42]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Penalize duplicate detections (low precision)',
                'clinical_scenario': 'Model outputs multiple boxes for single lesion',
                'edge_case_type': 'many_to_one'
            },
            {
                'id': 'multi_004',
                'category': 'multi_box',
                'name': 'One-to-Many (One Prediction, Multiple GT)',
                'description': 'Single prediction for multiple findings',
                'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.7], [0.2, 0.6, 0.4, 0.8]',
                'expected_behavior': 'Penalize missed findings (low recall)',
                'clinical_scenario': 'Model detects only one of three lesions',
                'edge_case_type': 'one_to_many'
            },
            {
                'id': 'multi_005',
                'category': 'multi_box',
                'name': 'Many-to-Many Complex',
                'description': 'Multiple predictions and multiple GT boxes',
                'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.52, 0.52, 0.62, 0.62]</answer>',
                'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]',
                'expected_behavior': 'Partial credit based on matches',
                'clinical_scenario': 'Complex case: some correct, some missed, some hallucinated',
                'edge_case_type': 'many_to_many'
            },

            # ================================================================
            # CATEGORY 4: GEOMETRIC EDGE CASES
            # ================================================================
            {
                'id': 'geo_001',
                'category': 'geometric',
                'name': 'Tiny Box (Small Finding)',
                'description': 'Very small bounding box',
                'prediction': '<answer>[0.45, 0.45, 0.47, 0.47]</answer>',
                'ground_truth': '[0.45, 0.45, 0.47, 0.47]',
                'expected_behavior': 'Should handle small boxes correctly',
                'clinical_scenario': 'Small nodule detection',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'geo_002',
                'category': 'geometric',
                'name': 'Large Box (Full Image)',
                'description': 'Box covers most of image',
                'prediction': '<answer>[0.05, 0.05, 0.95, 0.95]</answer>',
                'ground_truth': '[0.05, 0.05, 0.95, 0.95]',
                'expected_behavior': 'Should handle large boxes correctly',
                'clinical_scenario': 'Large infiltrate covering lung field',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'geo_003',
                'category': 'geometric',
                'name': 'Vertical Box (Tall and Narrow)',
                'description': 'High aspect ratio vertical box',
                'prediction': '<answer>[0.4, 0.1, 0.5, 0.9]</answer>',
                'ground_truth': '[0.4, 0.1, 0.5, 0.9]',
                'expected_behavior': 'Should handle aspect ratios correctly',
                'clinical_scenario': 'Elongated rib fracture',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'geo_004',
                'category': 'geometric',
                'name': 'Horizontal Box (Wide and Short)',
                'description': 'High aspect ratio horizontal box',
                'prediction': '<answer>[0.1, 0.45, 0.9, 0.55]</answer>',
                'ground_truth': '[0.1, 0.45, 0.9, 0.55]',
                'expected_behavior': 'Should handle aspect ratios correctly',
                'clinical_scenario': 'Wide pleural effusion line',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'geo_005',
                'category': 'geometric',
                'name': 'Corner Box (Edge of Image)',
                'description': 'Box at corner boundary',
                'prediction': '<answer>[0.8, 0.8, 1.0, 1.0]</answer>',
                'ground_truth': '[0.8, 0.8, 1.0, 1.0]',
                'expected_behavior': 'Should handle boundary cases',
                'clinical_scenario': 'Finding at image periphery',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'geo_006',
                'category': 'geometric',
                'name': 'Zero-Area Box (Degenerate)',
                'description': 'Box with zero area (x1=x2 or y1=y2)',
                'prediction': '<answer>[0.5, 0.5, 0.5, 0.5]</answer>',
                'ground_truth': '[0.5, 0.5, 0.7, 0.7]',
                'expected_behavior': 'Should handle gracefully (zero IoU)',
                'clinical_scenario': 'Malformed prediction',
                'edge_case_type': 'one_to_one'
            },

            # ================================================================
            # CATEGORY 5: EXTREME OVERLAPPING SCENARIOS
            # ================================================================
            {
                'id': 'overlap_001',
                'category': 'overlap',
                'name': 'Completely Nested Box',
                'description': 'Predicted box fully inside ground truth',
                'prediction': '<answer>[0.15, 0.25, 0.25, 0.35]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Moderate IoU due to size difference',
                'clinical_scenario': 'Conservative lesion boundary estimate',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'overlap_002',
                'category': 'overlap',
                'name': 'Fully Enclosing Box',
                'description': 'Predicted box fully contains ground truth',
                'prediction': '<answer>[0.05, 0.15, 0.35, 0.45]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Moderate IoU due to size difference',
                'clinical_scenario': 'Overly generous lesion boundary',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'overlap_003',
                'category': 'overlap',
                'name': 'Partial Overlap - Shifted Right',
                'description': 'Box shifted horizontally, partial overlap',
                'prediction': '<answer>[0.2, 0.2, 0.4, 0.4]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Moderate IoU (≈0.5)',
                'clinical_scenario': 'Horizontally misaligned detection',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'overlap_004',
                'category': 'overlap',
                'name': 'Diagonal Offset',
                'description': 'Box offset diagonally',
                'prediction': '<answer>[0.15, 0.25, 0.35, 0.45]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Lower IoU due to diagonal shift',
                'clinical_scenario': 'Diagonally misaligned detection',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'overlap_005',
                'category': 'overlap',
                'name': 'Minimal Overlap - Corner Touch',
                'description': 'Boxes barely overlap at corner',
                'prediction': '<answer>[0.25, 0.35, 0.45, 0.55]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Very low IoU, likely no match',
                'clinical_scenario': 'Severely mislocalized detection',
                'edge_case_type': 'one_to_one'
            },
            {
                'id': 'overlap_006',
                'category': 'overlap',
                'name': 'No Overlap - Completely Separate',
                'description': 'Boxes do not overlap at all',
                'prediction': '<answer>[0.6, 0.6, 0.8, 0.8]</answer>',
                'ground_truth': '[0.1, 0.2, 0.3, 0.4]',
                'expected_behavior': 'Zero IoU, no match',
                'clinical_scenario': 'Completely wrong location',
                'edge_case_type': 'one_to_one'
            },

            # ================================================================
            # CATEGORY 6: CHALLENGING CLINICAL SCENARIOS
            # ================================================================
            {
                'id': 'clinical_001',
                'category': 'clinical',
                'name': 'Multiple Small Nodules',
                'description': 'Several small findings',
                'prediction': '<answer>[0.1, 0.1, 0.15, 0.15], [0.3, 0.3, 0.35, 0.35], [0.6, 0.6, 0.65, 0.65]</answer>',
                'ground_truth': '[0.1, 0.1, 0.15, 0.15], [0.3, 0.3, 0.35, 0.35], [0.6, 0.6, 0.65, 0.65], [0.8, 0.8, 0.85, 0.85]',
                'expected_behavior': 'High precision, moderate recall (missed one)',
                'clinical_scenario': 'Multiple lung nodules, one missed',
                'edge_case_type': 'many_to_many'
            },
            {
                'id': 'clinical_002',
                'category': 'clinical',
                'name': 'Bilateral Findings',
                'description': 'Findings on both sides of image',
                'prediction': '<answer>[0.1, 0.4, 0.2, 0.6], [0.8, 0.4, 0.9, 0.6]</answer>',
                'ground_truth': '[0.1, 0.4, 0.2, 0.6], [0.8, 0.4, 0.9, 0.6]',
                'expected_behavior': 'Perfect detection of bilateral findings',
                'clinical_scenario': 'Bilateral pneumonia',
                'edge_case_type': 'many_to_many'
            },
            {
                'id': 'clinical_003',
                'category': 'clinical',
                'name': 'Overlapping Anatomical Structures',
                'description': 'Multiple overlapping boxes',
                'prediction': '<answer>[0.2, 0.2, 0.5, 0.5], [0.3, 0.3, 0.6, 0.6]</answer>',
                'ground_truth': '[0.2, 0.2, 0.5, 0.5], [0.3, 0.3, 0.6, 0.6]',
                'expected_behavior': 'Should handle overlapping boxes',
                'clinical_scenario': 'Overlapping pathologies',
                'edge_case_type': 'many_to_many'
            },

            # ================================================================
            # CATEGORY 7: MATCHING ALGORITHM STRESS TESTS
            # ================================================================
            {
                'id': 'matching_001',
                'category': 'matching',
                'name': 'Ambiguous Matches',
                'description': 'Two predictions could match two GT boxes',
                'prediction': '<answer>[0.1, 0.1, 0.25, 0.25], [0.15, 0.15, 0.3, 0.3]</answer>',
                'ground_truth': '[0.1, 0.1, 0.25, 0.25], [0.2, 0.2, 0.35, 0.35]',
                'expected_behavior': 'Greedy vs Hungarian may differ',
                'clinical_scenario': 'Close-proximity lesions with ambiguous matches',
                'edge_case_type': 'many_to_many'
            },
            {
                'id': 'matching_002',
                'category': 'matching',
                'name': 'Same IoU Multiple Matches',
                'description': 'Multiple GT boxes with same IoU to prediction',
                'prediction': '<answer>[0.2, 0.2, 0.3, 0.3]</answer>',
                'ground_truth': '[0.1, 0.1, 0.25, 0.25], [0.25, 0.25, 0.4, 0.4]',
                'expected_behavior': 'Tiebreaker depends on algorithm',
                'clinical_scenario': 'Equidistant lesions',
                'edge_case_type': 'one_to_many'
            },
            {
                'id': 'matching_003',
                'category': 'matching',
                'name': 'High Cardinality',
                'description': 'Many predictions and many GT boxes',
                'prediction': '<answer>' + ', '.join([f'[{0.1*i}, {0.1*i}, {0.1*i+0.08}, {0.1*i+0.08}]' for i in range(10)]) + '</answer>',
                'ground_truth': ', '.join([f'[{0.1*i}, {0.1*i}, {0.1*i+0.08}, {0.1*i+0.08}]' for i in range(10)]),
                'expected_behavior': 'Tests matching algorithm scalability',
                'clinical_scenario': 'Rare case: 10 separate findings',
                'edge_case_type': 'many_to_many'
            },
        ]

    def _categorize_tests(self) -> Dict[str, List[str]]:
        """Categorize test cases by type."""
        categories = {}
        for test in self.test_cases:
            cat = test['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test['id'])
        return categories

    def get_test_by_id(self, test_id: str) -> Dict[str, Any]:
        """Get a specific test case by ID."""
        for test in self.test_cases:
            if test['id'] == test_id:
                return test
        raise ValueError(f"Test ID '{test_id}' not found")

    def get_tests_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all test cases in a category."""
        return [test for test in self.test_cases if test['category'] == category]

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Get all test cases."""
        return self.test_cases

    def get_category_summary(self) -> Dict[str, int]:
        """Get count of tests per category."""
        summary = {}
        for cat in self.categories:
            summary[cat] = len(self.categories[cat])
        return summary

    def print_summary(self):
        """Print a summary of the test suite."""
        print("=" * 80)
        print("EDGE CASE TEST SUITE SUMMARY")
        print("=" * 80)
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"\nTests by category:")
        for cat, count in self.get_category_summary().items():
            print(f"  {cat:20s}: {count:3d} tests")
        print("=" * 80)


def main():
    """Run test suite summary."""
    suite = EdgeCaseTestSuite()
    suite.print_summary()

    print("\n" + "=" * 80)
    print("DETAILED TEST CASES")
    print("=" * 80)

    for category in suite.categories.keys():
        tests = suite.get_tests_by_category(category)
        print(f"\n{category.upper()} ({len(tests)} tests)")
        print("-" * 80)
        for test in tests:
            print(f"  [{test['id']}] {test['name']}")
            print(f"      {test['description']}")
            print(f"      Clinical: {test['clinical_scenario']}")


if __name__ == "__main__":
    main()
