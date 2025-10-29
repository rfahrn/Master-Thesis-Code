# GPRO Reward Functions - Complete Documentation Index

## Overview

This repository contains **5 reward functions** (R1-R5) for training Vision-Language Models (VLMs) on radiology grounding/detection tasks using GRPO (Group Relative Policy Optimization). All functions are organized by complexity, from simplest (R1) to most advanced (R5).

---

## Documentation Files (Start Here!)

### 1. **QUICK_REFERENCE_GUIDE.md** ← START HERE
- **What:** 30-second summaries, decision tree, hyperparameter tuning
- **Best for:** Quick lookups while coding
- **Size:** ~10 KB
- **Read time:** 5-10 minutes
- **Key sections:**
  - At-a-glance comparison table
  - Which function to use (decision tree)
  - Common utilities used by all functions
  - Hyperparameter tuning guide
  - Expected performance ranges

### 2. **ARCHITECTURE_OVERVIEW.txt** ← FOR VISUAL LEARNERS
- **What:** Flowcharts, hierarchies, visual diagrams
- **Best for:** Understanding relationships and flow
- **Size:** ~19 KB
- **Read time:** 10-15 minutes
- **Key sections:**
  - GRPO training pipeline diagram
  - Reward function hierarchy (by complexity)
  - Matching strategies comparison (greedy vs Hungarian)
  - Mathematical core formulas
  - Performance comparison table
  - Decision tree for function selection
  - Curriculum learning strategy

### 3. **REWARD_FUNCTIONS_DETAILED_SUMMARY.md** ← FOR DEEP UNDERSTANDING
- **What:** Comprehensive technical analysis
- **Best for:** Understanding all details before implementing
- **Size:** ~26 KB
- **Read time:** 30-45 minutes
- **Key sections:**
  - All 5 functions analyzed in detail
  - Common utilities (extract_boxes, compute_iou, etc.)
  - Mathematical approaches for each function
  - Bounding box matching strategies
  - Evaluation framework overview
  - Integration guidelines with GRPO

### 4. **REWARD_FUNCTION_ANALYSIS.md** ← FOR EVALUATION RESULTS
- **What:** Empirical evaluation on 15 test scenarios
- **Best for:** Understanding which function performs best
- **Size:** ~14 KB
- **Read time:** 20-30 minutes
- **Key sections:**
  - Performance rankings (R1-R5)
  - Test scenario descriptions
  - Detailed performance analysis
  - Critical insights for GRPO training
  - Recommendations by training phase
  - Citation format for thesis

### 5. **Reward_Functions/README.md** ← IN-REPO GUIDE
- **What:** Function-specific guides with formulas
- **Best for:** Understanding each R1-R5 function
- **Size:** ~7 KB
- **Read time:** 15-20 minutes
- **Key sections:**
  - Overview of all 5 functions
  - Ordered by complexity
  - Formulas and hyperparameters
  - Recommendations for GRPO
  - Hyperparameter tuning guide

---

## Reward Functions Summary

| Function | File | Approach | Complexity | When to Use |
|----------|------|----------|-----------|-------------|
| **R1** | `R1.py` | Binary AP@0.5 threshold | Simplest | Baseline/COCO matching |
| **R2** | `R2.py` | F-β × mean_IoU (greedy) | Simple | General GRPO baseline |
| **R3** | `R3.py` | F-β × mean_IoU (clean) | Simple | Production (safest choice) |
| **R4** | `R4.py` | F-β × smooth_spline | Medium | Medical/advanced GRPO |
| **R5** | `R5.py` | F-β + Hungarian matching | Complex | Exploration/strict medical |

---

## File Structure

```
Master-Thesis-Code/
├── Reward_Functions/
│   ├── R1.py                     # Simple AP@0.5
│   ├── R2.py                     # F-β × IoU (RL-optimized)
│   ├── R3.py                     # F-β × mean_IoU (production)
│   ├── R4.py                     # Enhanced smooth gradients
│   ├── R5.py                     # Strict medical (Hungarian)
│   └── README.md                 # In-folder guide
│
├── Documentation (READ THESE FIRST):
│   ├── INDEX_REWARD_FUNCTIONS.md         (this file)
│   ├── QUICK_REFERENCE_GUIDE.md          ← START HERE for quick lookup
│   ├── ARCHITECTURE_OVERVIEW.txt         ← START HERE for visual overview
│   ├── REWARD_FUNCTIONS_DETAILED_SUMMARY.md  ← For full understanding
│   └── REWARD_FUNCTION_ANALYSIS.md       ← For empirical results
│
├── Evaluation & Testing:
│   ├── evaluate_reward_functions.py  # Compare all functions
│   ├── test_r1_simple.py             # Simple standalone test
│   └── evaluation_output.log          # Test results
│
└── Integration:
    ├── grpo_simulation.py            # GRPO integration example
    └── curriculum.sh                 # Shell script for phases
```

---

## Quick Decision Guide

### I'm starting fresh. What should I read?

1. **First (5 min):** Read `QUICK_REFERENCE_GUIDE.md` section "The 30-Second Summary"
2. **Next (15 min):** Skim `ARCHITECTURE_OVERVIEW.txt` to see the big picture
3. **Then (30 min):** Read the relevant section of `REWARD_FUNCTIONS_DETAILED_SUMMARY.md`
4. **Finally (10 min):** Look up hyperparameters in `Reward_Functions/README.md`

### I need to choose which function to use

→ Go to **QUICK_REFERENCE_GUIDE.md** section "Which Should I Use?"

### I need to understand the mathematics

→ Go to **REWARD_FUNCTIONS_DETAILED_SUMMARY.md** section "Mathematical Approaches"

### I need to see performance comparisons

→ Go to **REWARD_FUNCTION_ANALYSIS.md** or **ARCHITECTURE_OVERVIEW.txt** performance table

### I need to implement one right now

→ Go to **Reward_Functions/README.md** and pick your function

### I'm writing a thesis

→ Go to **REWARD_FUNCTION_ANALYSIS.md** section "Citation for Thesis"

---

## Reading Paths by Use Case

### Path 1: "I just want to get training working fast"
```
1. QUICK_REFERENCE_GUIDE.md (5 min)
   → Section: "Which Should I Use?" → Choose R3
   → Section: "Standard Interface" → See how to call it
2. Reward_Functions/R3.py (2 min)
   → Copy the file, use compute_score() function
Done! Training starts.
```

### Path 2: "I need the best medical imaging setup"
```
1. QUICK_REFERENCE_GUIDE.md (10 min)
   → Section: "For Medical Imaging" → R4 recommended
2. ARCHITECTURE_OVERVIEW.txt (15 min)
   → Section: "Decision Tree" → Confirms R4
3. REWARD_FUNCTIONS_DETAILED_SUMMARY.md (30 min)
   → Section: "R4: Enhanced Smooth Gradients"
   → Section: "Integration with GRPO"
4. Reward_Functions/R4.py (5 min)
   → Review hyperparameters and usage
Ready to train with R4!
```

### Path 3: "I want to understand everything"
```
1. ARCHITECTURE_OVERVIEW.txt (20 min)
   → Full overview with diagrams
2. QUICK_REFERENCE_GUIDE.md (10 min)
   → Decision tree and comparisons
3. REWARD_FUNCTIONS_DETAILED_SUMMARY.md (60 min)
   → Every function explained in depth
4. REWARD_FUNCTION_ANALYSIS.md (30 min)
   → Empirical evaluation results
5. Reward_Functions/README.md (15 min)
   → All 5 functions quick reference
Complete expert understanding!
```

### Path 4: "I need to write a thesis"
```
1. QUICK_REFERENCE_GUIDE.md (10 min)
   → Section: "Summary Table"
2. REWARD_FUNCTION_ANALYSIS.md (30 min)
   → Full empirical analysis
   → Section: "Citation for Thesis" (copy-paste ready!)
3. REWARD_FUNCTIONS_DETAILED_SUMMARY.md (45 min)
   → Mathematical formulations section
4. Reward_Functions/README.md (10 min)
   → Hyperparameter tuning guide
5. ARCHITECTURE_OVERVIEW.txt (10 min)
   → For diagrams in thesis
Thesis content ready!
```

---

## Key Recommendations

### Single Best Choice
**R4 (Enhanced Smooth Gradients)** - provides the best balance:
- Good performance (mean=0.4896)
- Smooth gradient shaping (better for RL)
- Strict on false negatives (medical safety)
- Manageable complexity (4-5 hyperparameters)

### For Different Scenarios

| Scenario | Recommended | Why |
|----------|------------|-----|
| Baseline/Reference | R1 | Matches COCO AP@0.5 metric |
| General GRPO training | R3 | Simple, stable, proven |
| Medical imaging | R4 | Strict on false negatives |
| Exploration phase | R5 | Best for complex multi-box |
| Maximum stability | R3 or R6 | Lowest variance |
| Learning speed | R5 | Highest mean reward |

### Curriculum Learning
```
Phase 1 (0-20%):   R5 with MIN_IOU=0.2  → Exploration
Phase 2 (20-70%):  R4 with MIN_IOU=0.5  → Refinement  
Phase 3 (70-100%): R3 with MIN_IOU=0.5  → Optimization
```

---

## Common Questions Answered

### Q: Which function should I start with?
**A:** R3. It's simple (3 parameters), fast (greedy matching), and proven. If you need medical imaging safety, use R4 instead.

### Q: What's the difference between R2 and R3?
**A:** Functionally identical. R3 has cleaner documentation and emphasizes why it's good for RL. Use R3.

### Q: Why is R4 recommended over R3?
**A:** R4 adds smooth gradient shaping (via cubic spline) which improves learning stability. Also, it's stricter on false negatives (medical safety). Only 1 extra hyperparameter.

### Q: When should I use R5?
**A:** 
- Early training/exploration (partial credit helps)
- Complex multi-box scenarios (performs best)
- Medical applications requiring zero false positives
- If global optimality matters (Hungarian matching)

### Q: Why do some functions have more hyperparameters?
**A:** More hyperparameters = more control but harder to tune. Start simple (R3), add complexity (R4) if needed.

### Q: Can I switch functions during training?
**A:** Yes! Curriculum learning (R5→R4→R3) often gives best results. Start with exploration (R5), move to refinement (R4), finish with stability (R3).

### Q: What's the performance difference in practice?
**A:** Very small for most cases (all ~0.48-0.49 mean reward). Differences matter in edge cases:
- Multi-box scenarios: R5 significantly better (0.67 vs 0.56)
- False negatives: R4 most strict (0.27 vs 0.33)
- Stability: R3 most stable (0.369 std)

### Q: Do I need scipy for R4?
**A:** Yes, for the cubic spline. But scipy is a standard library, so not a problem.

### Q: Can I use Hungarian matching (R5) instead of greedy (R3/R4)?
**A:** Yes, but it's slower (O(n³) vs O(n² log n)). Usually not worth it for RL training unless global optimality is critical.

---

## Testing Your Setup

### Quick test (no dependencies)
```bash
cd /home/user/Master-Thesis-Code
python test_r1_simple.py
# Tests R1 without numpy/matplotlib
```

### Test individual function
```bash
python Reward_Functions/R3.py
# Tests R3 with various scenarios
```

### Compare all functions
```bash
python evaluate_reward_functions.py
# Compares R3, R4, R5, R6 on 15 test cases
# Generates plots and statistics
```

---

## Integration Checklist

- [ ] Read QUICK_REFERENCE_GUIDE.md (5 min)
- [ ] Choose a function (R3 recommended)
- [ ] Read function's section in DETAILED_SUMMARY.md
- [ ] Review hyperparameters in Reward_Functions/README.md
- [ ] Test with test_r1_simple.py or evaluate_reward_functions.py
- [ ] Integrate into GRPO training loop using compute_score()
- [ ] Monitor reward curves during training
- [ ] Document which function and hyperparameters you used

---

## For Thesis Writers

### Recommended Structure

**Methods Section:**
- Reference: REWARD_FUNCTION_ANALYSIS.md → "Citation for Thesis"
- Include: "We evaluated reward functions R3-R5 across 15 diverse test scenarios..."

**Results Section:**
- Use data from: REWARD_FUNCTION_ANALYSIS.md → "Detailed Performance Analysis"
- Include tables: ARCHITECTURE_OVERVIEW.txt → "Performance Comparison"

**Discussion Section:**
- Cite recommendation: R4 for medical, R3 for general, R5 for exploration
- Reference: REWARD_FUNCTION_ANALYSIS.md → "Recommendations by Training Phase"

### Key Metrics to Report
- Mean reward: ~0.48-0.49
- Standard deviation (stability): ~0.37
- Multi-box performance: R5 significantly better
- False negative handling: R4 most strict

---

## Additional Resources

### Evaluation Results
- `REWARD_FUNCTION_ANALYSIS.md` - Complete empirical evaluation on 15 scenarios
- `evaluation_output.log` - Raw test output

### In-Folder Guides
- `Reward_Functions/README.md` - Per-function guide with all formulas

### Integration Examples
- `grpo_simulation.py` - Example GRPO training loop
- `curriculum.sh` - Shell script for curriculum learning

---

## File Sizes and Read Times

| File | Size | Time | Purpose |
|------|------|------|---------|
| QUICK_REFERENCE_GUIDE.md | 10 KB | 5-10 min | Quick lookup |
| ARCHITECTURE_OVERVIEW.txt | 19 KB | 10-15 min | Visual overview |
| REWARD_FUNCTIONS_DETAILED_SUMMARY.md | 26 KB | 30-45 min | Full understanding |
| REWARD_FUNCTION_ANALYSIS.md | 14 KB | 20-30 min | Evaluation results |
| Reward_Functions/README.md | 7 KB | 10-15 min | Function references |

**Total reading time:** 90-140 minutes for full understanding
**Minimum reading time:** 5-10 minutes for quick start

---

## Getting Help

### "I don't understand X"
1. Check QUICK_REFERENCE_GUIDE.md for quick answers
2. Look at ARCHITECTURE_OVERVIEW.txt for visual explanation
3. Read detailed section in REWARD_FUNCTIONS_DETAILED_SUMMARY.md

### "I need to modify a function"
1. Understand the core formula (section "Mathematical Approaches")
2. Check hyperparameters in Reward_Functions/README.md
3. Test changes with evaluate_reward_functions.py

### "Something isn't working"
1. Check test output: `python Reward_Functions/R3.py`
2. Verify input format in QUICK_REFERENCE_GUIDE.md section "Input/Output Formats"
3. Enable `return_details=True` to debug

### "I want to write about this"
→ Copy-paste ready text: REWARD_FUNCTION_ANALYSIS.md section "Citation for Thesis"

---

## Summary

You have access to **5 reward functions** organized by complexity:
- **R1**: Simplest (AP@0.5)
- **R2/R3**: Simple (F-β × IoU)
- **R4**: Medium (smooth gradients) ← **RECOMMENDED**
- **R5**: Complex (Hungarian matching)

All are documented with:
- Quick reference guides
- Detailed technical analysis
- Empirical evaluation results
- Visual architecture diagrams
- Integration examples
- Thesis citation formats

**Start with:** QUICK_REFERENCE_GUIDE.md (5 minutes)
**Then use:** Function-specific sections as needed

Good luck with your GRPO training!

---

**Generated:** October 2025
**Framework:** Master Thesis GPRO Reward Functions
**Status:** All functions documented, tested, and ready to use

