

# Mathematical Definitions of GPRO Reward Functions

**For Master Thesis: Comparative Analysis of RL Reward Functions for Radiology Grounding**

---

## Table of Contents

1. [Common Notation and Definitions](#1-common-notation-and-definitions)
2. [Shared Components](#2-shared-components)
3. [R1: Average Precision at IoU 0.5](#3-r1-average-precision-at-iou-05)
4. [R2: F-beta × IoU Baseline](#4-r2-f-beta--iou-baseline)
5. [R3: F-beta × mean_IoU](#5-r3-f-beta--mean_iou)
6. [R4: Enhanced Smooth Spline](#6-r4-enhanced-smooth-spline)
7. [R5: Strict Medical Grounding](#7-r5-strict-medical-grounding)
8. [Comparative Analysis](#8-comparative-analysis)

---

## 1. Common Notation and Definitions

### 1.1 Basic Notation

| Symbol | Description |
|--------|-------------|
| $\mathcal{P} = \{p_1, p_2, \ldots, p_n\}$ | Set of predicted bounding boxes |
| $\mathcal{G} = \{g_1, g_2, \ldots, g_m\}$ | Set of ground truth bounding boxes |
| $n = |\mathcal{P}|$ | Number of predicted boxes |
| $m = |\mathcal{G}|$ | Number of ground truth boxes |
| $b = [x_1, y_1, x_2, y_2]$ | Bounding box coordinates |
| $\text{IoU}(p_i, g_j)$ | Intersection over Union between boxes $p_i$ and $g_j$ |
| $\tau$ | IoU matching threshold (typically 0.5) |
| $\beta$ | F-beta score parameter (controls precision-recall trade-off) |

### 1.2 Evaluation Metrics

#### True Positives (TP)
Number of predicted boxes correctly matched to ground truth boxes above IoU threshold $\tau$.

$$\text{TP} = |\{p_i \in \mathcal{P} : \exists g_j \in \mathcal{G}, \text{IoU}(p_i, g_j) \geq \tau \text{ and } g_j \text{ unmatched}\}|$$

#### False Positives (FP)
Number of predicted boxes that fail to match any ground truth box.

$$\text{FP} = |\{p_i \in \mathcal{P} : \forall g_j \in \mathcal{G}, \text{IoU}(p_i, g_j) < \tau \text{ or } g_j \text{ already matched}\}|$$

#### False Negatives (FN)
Number of ground truth boxes that remain unmatched after matching process.

$$\text{FN} = |\{g_j \in \mathcal{G} : g_j \text{ unmatched after matching}\}|$$

#### Precision and Recall

$$P = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{TP}}{n}$$

$$R = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{m}$$

#### F-beta Score

$$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

Where:
- $\beta = 1$: Balanced F1 score (equal weight to precision and recall)
- $\beta > 1$: Emphasizes recall (fewer false negatives)
- $\beta < 1$: Emphasizes precision (fewer false positives)

---

## 2. Shared Components

### 2.1 Intersection over Union (IoU)

For two bounding boxes $b_1 = [x_1^{(1)}, y_1^{(1)}, x_2^{(1)}, y_2^{(1)}]$ and $b_2 = [x_1^{(2)}, y_1^{(2)}, x_2^{(2)}, y_2^{(2)}]$:

#### Intersection Area

$$\text{Inter}(b_1, b_2) = \max(0, x_I) \cdot \max(0, y_I)$$

where:

$$x_I = \min(x_2^{(1)}, x_2^{(2)}) - \max(x_1^{(1)}, x_1^{(2)})$$

$$y_I = \min(y_2^{(1)}, y_2^{(2)}) - \max(y_1^{(1)}, y_1^{(2)})$$

#### Union Area

$$\text{Union}(b_1, b_2) = A(b_1) + A(b_2) - \text{Inter}(b_1, b_2)$$

where $A(b) = (x_2 - x_1)(y_2 - y_1)$ is the area of box $b$.

#### IoU Definition

$$\text{IoU}(b_1, b_2) = \begin{cases}
\frac{\text{Inter}(b_1, b_2)}{\text{Union}(b_1, b_2)} & \text{if } \text{Union}(b_1, b_2) > 0 \\
0 & \text{otherwise}
\end{cases}$$

**Properties:**
- $\text{IoU}(b_1, b_2) \in [0, 1]$
- $\text{IoU}(b_1, b_2) = 1$ iff $b_1 = b_2$ (perfect match)
- $\text{IoU}(b_1, b_2) = 0$ iff boxes do not overlap
- Symmetric: $\text{IoU}(b_1, b_2) = \text{IoU}(b_2, b_1)$

### 2.2 Box Matching Algorithms

#### Greedy Matching Algorithm

```
Input: P = {p₁, ..., pₙ}, G = {g₁, ..., gₘ}, threshold τ
Output: Set of matches M ⊆ P × G

1. Compute IoU matrix: IoU[i,j] = IoU(pᵢ, gⱼ) for all i,j
2. Initialize: M = ∅, matched_gt = ∅
3. Sort predictions by max IoU: indices = argsort(-max_j IoU[i,j])
4. For each i in indices:
     a. Find best unmatched GT: j* = argmax_{j∉matched_gt} IoU[i,j]
     b. If IoU[i,j*] ≥ τ:
          - Add (i, j*) to M
          - Add j* to matched_gt
5. Return M
```

**Complexity:** $O(nm \log n)$ where $n = |\mathcal{P}|$, $m = |\mathcal{G}|$

**Properties:**
- Fast and deterministic
- Greedy: may not find globally optimal matching
- Prioritizes high-confidence predictions (sorted by max IoU)

#### Hungarian Matching Algorithm

```
Input: P = {p₁, ..., pₙ}, G = {g₁, ..., gₘ}, threshold τ
Output: Optimal matching M ⊆ P × G

1. Compute cost matrix: C[i,j] = 1 - IoU(pᵢ, gⱼ) for all i,j
2. Apply Hungarian algorithm to find optimal assignment minimizing total cost
3. Filter matches: M = {(i,j) : (i,j) in assignment and IoU[i,j] ≥ τ}
4. Return M
```

**Complexity:** $O((n+m)^3)$ using Kuhn-Munkres algorithm

**Properties:**
- Finds globally optimal matching
- Guaranteed to minimize total cost (or maximize total IoU)
- More computationally expensive than greedy
- Particularly important when predictions have similar IoU scores to multiple GT boxes

---

## 3. R1: Average Precision at IoU 0.5

### 3.1 Mathematical Definition

R1 computes Average Precision (AP) at a fixed IoU threshold of 0.5, following the COCO evaluation protocol.

#### Reward Function

$$R_1(\mathcal{P}, \mathcal{G}) = \begin{cases}
\delta_{\text{no-box}} & \text{if } n = 0 \text{ and } m = 0 \\
\text{AP}_{0.5}(\mathcal{P}, \mathcal{G}) & \text{otherwise}
\end{cases}$$

where $\delta_{\text{no-box}} = 0.2$ is a small bonus for correct negative predictions.

#### Average Precision Computation

Given a set of predictions $\mathcal{P}$ and ground truths $\mathcal{G}$:

1. **Match boxes** using greedy matching with $\tau = 0.5$

2. **Assign TP/FP labels** to each prediction $p_i$:
   $$\text{label}(p_i) = \begin{cases}
   \text{TP} & \text{if } p_i \text{ matched to some } g_j \\
   \text{FP} & \text{otherwise}
   \end{cases}$$

3. **Compute cumulative precision and recall**:
   $$P_k = \frac{\sum_{i=1}^k \mathbb{1}[\text{label}(p_i) = \text{TP}]}{k}$$

   $$R_k = \frac{\sum_{i=1}^k \mathbb{1}[\text{label}(p_i) = \text{TP}]}{m}$$

4. **Interpolate precision** (monotonically decreasing):
   $$\tilde{P}(R) = \max_{R' \geq R} P(R')$$

5. **Compute Average Precision**:
   $$\text{AP}_{0.5} = \sum_{k=1}^n \tilde{P}(R_k) \cdot (R_k - R_{k-1})$$

   Equivalently:
   $$\text{AP}_{0.5} = \int_0^1 \tilde{P}(R) \, dR$$

### 3.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\tau$ | 0.5 | IoU matching threshold |
| $\delta_{\text{no-box}}$ | 0.2 | True negative bonus |

### 3.3 Properties

- **Range:** $[0, 1]$ (or $\delta_{\text{no-box}}$ for correct negatives)
- **Threshold behavior:** Hard cutoff at IoU = 0.5
- **Multi-box handling:** Naturally handles multiple predictions and ground truths
- **Monotonicity:** Higher IoU → higher chance of counting as TP → higher AP
- **Gradient:** Discontinuous at IoU = 0.5 threshold

### 3.4 Edge Case Behavior

| Scenario | $n$ | $m$ | Reward |
|----------|-----|-----|--------|
| True Negative | 0 | 0 | $\delta_{\text{no-box}} = 0.2$ |
| Hallucination | $>0$ | 0 | $0$ (all FP) |
| Missed Detection | 0 | $>0$ | $0$ (all FN) |
| Perfect Match | 1 | 1 | $1.0$ (if IoU ≥ 0.5) |
| Partial Match | $k$ | $m$ | $\frac{k}{k+m-k} \leq \text{AP} \leq \frac{k}{m}$ |

---

## 4. R2: F-beta × IoU Baseline

### 4.1 Mathematical Definition

R2 uses F-beta score multiplied by the product of individual matched IoU values.

#### Reward Function

$$R_2(\mathcal{P}, \mathcal{G}) = \begin{cases}
\delta_{\text{no-box}} & \text{if } n = 0 \text{ and } m = 0 \\
0 & \text{if } n = 0 \text{ or } m = 0 \\
F_\beta \times \prod_{(i,j) \in \mathcal{M}} \text{IoU}(p_i, g_j) & \text{otherwise}
\end{cases}$$

where:
- $\mathcal{M}$ is the set of matches from greedy matching
- $F_\beta$ is computed from TP, FP, FN counts

#### Step-by-Step Computation

1. **Perform greedy matching** with threshold $\tau = 0.5$ → get matches $\mathcal{M}$

2. **Count TP, FP, FN**:
   $$\text{TP} = |\mathcal{M}|$$
   $$\text{FP} = n - |\mathcal{M}|$$
   $$\text{FN} = m - |\mathcal{M}|$$

3. **Compute F-beta**:
   $$P = \frac{\text{TP}}{n}, \quad R = \frac{\text{TP}}{m}$$
   $$F_\beta = \frac{(1 + \beta^2) P R}{\beta^2 P + R}$$

4. **Compute IoU product** (geometric mean quality):
   $$Q = \prod_{(i,j) \in \mathcal{M}} \text{IoU}(p_i, g_j)$$

5. **Final reward**:
   $$R_2 = F_\beta \times Q$$

### 4.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 1.5 | F-beta parameter (mild recall emphasis) |
| $\tau$ | 0.5 | IoU matching threshold |
| $\delta_{\text{no-box}}$ | 0.2 | True negative bonus |

### 4.3 Properties

- **Range:** $[0, 1]$
- **Detection component:** $F_\beta$ measures cardinality correctness
- **Localization component:** IoU product measures quality of matches
- **Multiplicative:** Both detection and localization must be good for high reward
- **Gradient:** Product of IoUs creates very steep penalty for poor matches
- **Sensitivity:** Highly sensitive to even one poor-quality match (product goes to zero)

### 4.4 Limitations

The product $\prod \text{IoU}$ has **exponential decay** with number of matches:

$$Q = \prod_{k=1}^K \text{IoU}_k \approx \text{IoU}_{\text{avg}}^K$$

For $K$ matches with average IoU = 0.7:
- $K=1$: $Q = 0.70$
- $K=2$: $Q = 0.49$
- $K=3$: $Q = 0.34$
- $K=5$: $Q = 0.17$

This makes R2 **unsuitable for multi-box scenarios**.

---

## 5. R3: F-beta × mean_IoU

### 5.1 Mathematical Definition

R3 addresses R2's multi-box limitation by using arithmetic mean instead of product.

#### Reward Function

$$R_3(\mathcal{P}, \mathcal{G}) = \begin{cases}
\delta_{\text{no-box}} & \text{if } n = 0 \text{ and } m = 0 \\
0 & \text{if } n = 0 \text{ or } m = 0 \\
F_\beta \times \overline{\text{IoU}} & \text{otherwise}
\end{cases}$$

where:

$$\overline{\text{IoU}} = \begin{cases}
\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \text{IoU}(p_i, g_j) & \text{if } |\mathcal{M}| > 0 \\
0 & \text{otherwise}
\end{cases}$$

#### Step-by-Step Computation

1. **Perform greedy matching** with threshold $\tau = 0.5$ → get matches $\mathcal{M}$

2. **Compute TP, FP, FN and F-beta** (same as R2)

3. **Compute mean IoU**:
   $$\overline{\text{IoU}} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \text{IoU}(p_i, g_j)$$

4. **Final reward**:
   $$R_3 = F_\beta \times \overline{\text{IoU}}$$

### 5.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 1.5 | F-beta parameter (mild recall emphasis) |
| $\tau$ | 0.5 | IoU matching threshold |
| $\delta_{\text{no-box}}$ | 0.2 | True negative bonus |

### 5.3 Properties

- **Range:** $[0, 1]$
- **Linear combination:** Arithmetic mean is linear in IoU values
- **Multi-box friendly:** No exponential decay with number of matches
- **Balanced:** $F_\beta$ weighs cardinality, $\overline{\text{IoU}}$ weighs quality
- **Stable:** Mean IoU remains in reasonable range even with many boxes
- **Production-ready:** Used in deployment due to simplicity and stability

### 5.4 Comparison with R2

For $K$ matches with IoUs $\{\text{IoU}_1, \ldots, \text{IoU}_K\}$:

**R2 (product):**
$$Q_{\text{R2}} = \prod_{k=1}^K \text{IoU}_k$$

**R3 (mean):**
$$Q_{\text{R3}} = \frac{1}{K} \sum_{k=1}^K \text{IoU}_k$$

By AM-GM inequality:

$$Q_{\text{R3}} \geq Q_{\text{R2}}^{1/K}$$

with equality only when all IoUs are equal.

**Example:** IoUs = [0.8, 0.7, 0.6]
- R2: $Q = 0.8 \times 0.7 \times 0.6 = 0.336$
- R3: $Q = (0.8 + 0.7 + 0.6)/3 = 0.700$

R3 is **much more generous** in multi-box scenarios.

---

## 6. R4: Enhanced Smooth Spline

### 6.1 Mathematical Definition

R4 enhances R3 by applying a smooth spline transformation to IoU values for better gradient properties.

#### Reward Function

$$R_4(\mathcal{P}, \mathcal{G}) = \begin{cases}
\delta_{\text{no-box}} & \text{if } n = 0 \text{ and } m = 0 \\
0 & \text{if } n = 0 \text{ or } m = 0 \\
F_\beta \times Q_{\text{smooth}} & \text{otherwise}
\end{cases}$$

where:

$$Q_{\text{smooth}} = \begin{cases}
\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} s(\text{IoU}(p_i, g_j)) & \text{if } |\mathcal{M}| > 0 \\
0 & \text{otherwise}
\end{cases}$$

and $s: [0,1] \to [0,1]$ is a smooth spline function.

#### Cubic Hermite Spline Definition

The spline $s(\text{IoU})$ is defined by 4 control points:

| IoU | Reward | Gradient |
|-----|--------|----------|
| 0.0 | 0.0 | 1.5 |
| 0.5 | 0.5 | 1.1 |
| 0.8 | 0.8 | 0.9 |
| 1.0 | 1.0 | 0.5 |

The cubic Hermite spline interpolates these points with prescribed derivatives.

For an interval $[x_k, x_{k+1}]$ with values $(y_k, y_{k+1})$ and derivatives $(m_k, m_{k+1})$:

$$s(x) = h_{00}(t) y_k + h_{10}(t) h m_k + h_{01}(t) y_{k+1} + h_{11}(t) h m_{k+1}$$

where:
- $t = \frac{x - x_k}{h}$, $h = x_{k+1} - x_k$
- $h_{00}(t) = 2t^3 - 3t^2 + 1$ (Hermite basis function)
- $h_{10}(t) = t^3 - 2t^2 + t$
- $h_{01}(t) = -2t^3 + 3t^2$
- $h_{11}(t) = t^3 - t^2$

#### Design Rationale

The gradient values are chosen to provide:

1. **Steep gradient [0, 0.5]:** Encourages crossing the matching threshold
2. **Moderate gradient [0.5, 0.8]:** Steady improvement in "good" range
3. **Gentle gradient [0.8, 1.0]:** Diminishing returns for near-perfect boxes

This creates **better learning dynamics** than linear mean IoU.

### 6.2 Optional Center-Aware Enhancement

R4 optionally includes a center distance penalty:

$$Q_{\text{center-aware}} = (1 - w_c) Q_{\text{smooth}} + w_c Q_{\text{center}}$$

where:

$$Q_{\text{center}} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \left(1 - d_{\text{center}}(p_i, g_j)\right)$$

and the normalized center distance is:

$$d_{\text{center}}(p_i, g_j) = \frac{\|c(p_i) - c(g_j)\|_2}{\text{diag}(g_j)}$$

where:
- $c(b) = \left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right)$ is the box center
- $\text{diag}(b) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$ is the box diagonal

### 6.3 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 1.5 | F-beta parameter |
| $\tau$ | 0.5 | IoU matching threshold |
| $\delta_{\text{no-box}}$ | 0.2 | True negative bonus |
| $w_c$ | 0.15 | Center-aware weight (if enabled) |
| USE_CENTER_AWARE | False | Enable center distance refinement |

### 6.4 Properties

- **Range:** $[0, 1]$
- **Smoothness:** $C^1$ continuous (continuous first derivative)
- **Monotonicity:** $s(\text{IoU})$ is monotonically increasing
- **Better gradients:** Steeper where it matters, gentler at extremes
- **RL-friendly:** Smooth reward shaping improves policy gradient learning

### 6.5 Gradient Analysis

Compare gradients of R3 vs R4:

**R3 (linear):**
$$\frac{\partial Q_{\text{R3}}}{\partial \text{IoU}} = \frac{1}{|\mathcal{M}|} \quad \text{(constant)}$$

**R4 (spline):**
$$\frac{\partial Q_{\text{R4}}}{\partial \text{IoU}} = \frac{1}{|\mathcal{M}|} \cdot s'(\text{IoU})$$

where $s'(\text{IoU})$ varies with IoU:
- At IoU = 0.0: $s'(0) = 1.5$ (steep!)
- At IoU = 0.5: $s'(0.5) = 1.1$ (moderate)
- At IoU = 0.8: $s'(0.8) = 0.9$ (gentle)
- At IoU = 1.0: $s'(1) = 0.5$ (very gentle)

This **adaptive gradient** provides better learning signal.

---

## 7. R5: Strict Medical Grounding

### 7.1 Mathematical Definition

R5 uses a piecewise linear quality function and Hungarian matching for optimal assignment.

#### Reward Function

$$R_5(\mathcal{P}, \mathcal{G}) = \begin{cases}
\delta_{\text{no-box}} & \text{if } n = 0 \text{ and } m = 0 \\
0 & \text{if } n = 0 \text{ or } m = 0 \\
F_\beta \times Q_{\text{piecewise}} & \text{otherwise}
\end{cases}$$

where:

$$Q_{\text{piecewise}} = \begin{cases}
\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} q(\text{IoU}(p_i, g_j)) & \text{if } |\mathcal{M}| > 0 \\
0 & \text{otherwise}
\end{cases}$$

and $q: [0,1] \to [0,1]$ is a piecewise linear quality function.

#### Piecewise Quality Function

$$q(\text{IoU}) = \begin{cases}
0 & \text{if } \text{IoU} < 0.3 \\
\frac{\text{IoU} - 0.3}{0.2} \times 0.3 & \text{if } 0.3 \leq \text{IoU} < 0.5 \\
0.3 + \frac{\text{IoU} - 0.5}{0.2} \times 0.4 & \text{if } 0.5 \leq \text{IoU} < 0.7 \\
0.7 + \frac{\text{IoU} - 0.7}{0.3} \times 0.3 & \text{if } 0.7 \leq \text{IoU} \leq 1.0
\end{cases}$$

Simplified:

$$q(\text{IoU}) = \begin{cases}
0 & \text{IoU} < 0.3 \\
1.5 \cdot (\text{IoU} - 0.3) & 0.3 \leq \text{IoU} < 0.5 \\
0.3 + 2.0 \cdot (\text{IoU} - 0.5) & 0.5 \leq \text{IoU} < 0.7 \\
0.7 + 1.0 \cdot (\text{IoU} - 0.7) & 0.7 \leq \text{IoU} \leq 1.0
\end{cases}$$

#### Control Points

| IoU | Quality $q(\text{IoU})$ | Gradient $q'(\text{IoU})$ |
|-----|-------------------------|---------------------------|
| 0.0-0.3 | 0.0 | 0.0 (flat) |
| 0.3 | 0.0 | 1.5 (steep jump) |
| 0.5 | 0.3 | 2.0 (steepest!) |
| 0.7 | 0.7 | 1.0 (moderate) |
| 1.0 | 1.0 | 1.0 (constant to end) |

#### Design Rationale

- **Dead zone [0, 0.3]:** Zero quality for very poor matches
- **Steep zone [0.3, 0.5]:** Rapid reward increase near threshold
- **Steepest zone [0.5, 0.7]:** Maximum gradient in "acceptable" range
- **Linear zone [0.7, 1.0]:** Constant gradient for high-quality matches

This is **stricter** than R3/R4, especially for poor matches.

### 7.2 Hungarian Matching

Unlike R1-R4 (greedy), R5 uses **optimal bipartite matching**:

1. **Compute cost matrix:**
   $$C_{ij} = 1 - \text{IoU}(p_i, g_j)$$

2. **Solve assignment problem:**
   $$\mathcal{M}^* = \arg\min_{\mathcal{M}} \sum_{(i,j) \in \mathcal{M}} C_{ij}$$

3. **Filter by threshold:**
   $$\mathcal{M}_{\text{final}} = \{(i,j) \in \mathcal{M}^* : \text{IoU}(p_i, g_j) \geq \tau\}$$

**Why Hungarian?**
- Finds **globally optimal** matching
- Important when multiple predictions compete for same GT box
- Better for medical applications where optimal matching is critical

**Complexity trade-off:**
- Greedy: $O(nm \log n)$
- Hungarian: $O((n+m)^3)$

### 7.3 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 1.5 | F-beta parameter |
| $\tau$ | 0.5 | IoU matching threshold |
| $\delta_{\text{no-box}}$ | 0.2 | True negative bonus |
| IoU breakpoints | [0.3, 0.5, 0.7] | Piecewise function knots |
| Quality breakpoints | [0.0, 0.3, 0.7, 1.0] | Piecewise function values |

### 7.4 Properties

- **Range:** $[0, 1]$
- **Strictness:** Harder to achieve high reward than R3/R4
- **Optimal matching:** Guaranteed globally best assignment
- **Medical focus:** Designed for high-stakes medical grounding
- **Piecewise linear:** Simple, interpretable, with controlled gradients

---

## 8. Comparative Analysis

### 8.1 Summary Table

| Function | Type | Matching | Complexity | Gradient | Multi-Box |
|----------|------|----------|------------|----------|-----------|
| **R1** | AP@0.5 | Greedy (sorted) | Simple (1 param) | Discontinuous at τ | Excellent |
| **R2** | $F_\beta \times \prod \text{IoU}$ | Greedy | Simple (3 params) | Continuous, steep decay | Poor |
| **R3** | $F_\beta \times \overline{\text{IoU}}$ | Greedy (sorted) | Simple (3 params) | Continuous, flat | Good |
| **R4** | $F_\beta \times s(\overline{\text{IoU}})$ | Greedy (sorted) | Moderate (4-5 params) | Smooth, adaptive | Good |
| **R5** | $F_\beta \times q(\overline{\text{IoU}})$ | Hungarian | Complex (5 params) | Piecewise, strict | Good |

### 8.2 Reward Range Comparison

For a single perfect match (IoU = 1.0, one box):

| Function | Reward | Reason |
|----------|--------|--------|
| R1 | 1.0 | AP = 1.0 (perfect precision and recall) |
| R2 | 1.0 | $F_{1.5} \times 1.0 = 1.0$ |
| R3 | 1.0 | $F_{1.5} \times 1.0 = 1.0$ |
| R4 | 1.0 | $F_{1.5} \times s(1.0) = 1.0$ |
| R5 | 1.0 | $F_{1.5} \times q(1.0) = 1.0$ |

For IoU = 0.6 (above threshold):

| Function | Approx. Reward | Reason |
|----------|----------------|--------|
| R1 | ~0.6-1.0 | Depends on precision-recall curve |
| R2 | ~0.5 | $F_{1.5} \times 0.6 \approx 0.83 \times 0.6$ |
| R3 | ~0.5 | $F_{1.5} \times 0.6 \approx 0.83 \times 0.6$ |
| R4 | ~0.55 | $F_{1.5} \times s(0.6) \approx 0.83 \times 0.65$ |
| R5 | ~0.46 | $F_{1.5} \times q(0.6) \approx 0.83 \times 0.55$ |

### 8.3 Multi-Box Scaling

For 3 perfect matches (3 boxes, all IoU = 0.8):

| Function | Quality Component | Final Reward |
|----------|-------------------|--------------|
| R1 | AP ≈ 1.0 | ~1.0 |
| R2 | $0.8^3 = 0.512$ | ~0.42 ⚠️ |
| R3 | $\frac{0.8+0.8+0.8}{3} = 0.8$ | ~0.67 |
| R4 | $\frac{s(0.8)+s(0.8)+s(0.8)}{3} \approx 0.82$ | ~0.68 |
| R5 | $\frac{q(0.8)+q(0.8)+q(0.8)}{3} \approx 0.8$ | ~0.67 |

**Observation:** R2 severely penalizes multi-box scenarios (exponential decay).

### 8.4 Gradient Comparison

At IoU = 0.4 (below threshold):

| Function | Gradient w.r.t. IoU |
|----------|---------------------|
| R1 | ~0 (below threshold) |
| R2 | $\frac{\partial}{\partial \text{IoU}} \text{IoU}^K$ |
| R3 | $\frac{F_\beta}{K}$ (constant) |
| R4 | $\frac{F_\beta}{K} \cdot s'(0.4) \approx 1.4$ |
| R5 | $\frac{F_\beta}{K} \cdot q'(0.4) = 1.5$ |

**Observation:** R4 and R5 have steeper gradients near threshold, encouraging threshold crossing.

### 8.5 Computational Complexity

For $n$ predictions and $m$ ground truths:

| Function | Matching | IoU Computation | Quality Computation | Total |
|----------|----------|-----------------|---------------------|-------|
| R1 | $O(nm \log n)$ | $O(nm)$ | $O(n \log n)$ | $O(nm \log n)$ |
| R2 | $O(nm \log n)$ | $O(nm)$ | $O(K)$ | $O(nm \log n)$ |
| R3 | $O(nm \log n)$ | $O(nm)$ | $O(K)$ | $O(nm \log n)$ |
| R4 | $O(nm \log n)$ | $O(nm)$ | $O(K)$ | $O(nm \log n)$ |
| R5 | $O((n+m)^3)$ ⚠️ | $O(nm)$ | $O(K)$ | $O((n+m)^3)$ |

where $K = \min(n, m)$ is the number of matches.

**Observation:** R5 is significantly slower due to Hungarian algorithm, but more accurate.

### 8.6 Recommendation Matrix

| Use Case | Best Function | Reason |
|----------|---------------|--------|
| **Fast prototyping** | R1 | Simple, standard metric |
| **Production deployment** | R3 | Stable, proven, fast |
| **RL training (GRPO/PPO)** | R4 | Best gradients for policy learning |
| **Medical applications** | R5 | Strict quality requirements, optimal matching |
| **Multi-box scenarios** | R3, R4, R5 | Avoid R2 (exponential decay) |
| **Single-box tasks** | R1, R3, R4 | All perform well |
| **Interpretability** | R1, R3 | Standard metrics |
| **Research/exploration** | R4, R5 | Novel components worth studying |

---

## 9. Implementation Notes

### 9.1 Numerical Stability

All functions should implement:

1. **Epsilon smoothing** for division:
   $$P = \frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}, \quad \epsilon = 10^{-8}$$

2. **Clipping** to prevent NaN:
   $$\text{IoU} = \text{clip}(\text{IoU}, 0, 1)$$
   $$\text{reward} = \text{clip}(\text{reward}, 0, 1)$$

3. **Special case handling:**
   - Empty predictions and empty GT: return $\delta_{\text{no-box}}$
   - Empty predictions OR empty GT: return $0$
   - Division by zero: return $0$

### 9.2 Backward Compatibility

When changing hyperparameters, ensure:

1. **Reward range** remains $[0, 1]$
2. **True negative bonus** is consistent across functions
3. **Threshold $\tau$** is standardized (typically 0.5)
4. **F-beta $\beta$** is consistent (typically 1.5 for medical tasks)

### 9.3 Testing Recommendations

Each reward function should be tested on:

1. **Empty cases:** $(n=0, m=0)$, $(n>0, m=0)$, $(n=0, m>0)$
2. **Perfect match:** Single box, IoU = 1.0
3. **Threshold cases:** IoU = 0.49, 0.50, 0.51
4. **Multi-box:** 2, 3, 5, 10 boxes
5. **Edge geometries:** Tiny boxes, large boxes, overlapping boxes
6. **Numeric edge cases:** Zero-area boxes, negative coordinates, out-of-bounds

---

## References

1. **COCO Evaluation:** Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
2. **F-beta Score:** van Rijsbergen, "Information Retrieval", 1979
3. **IoU Metric:** Everingham et al., "The PASCAL Visual Object Classes Challenge", IJCV 2010
4. **Hungarian Algorithm:** Kuhn, "The Hungarian Method for the assignment problem", Naval Research Logistics 1955
5. **Cubic Hermite Splines:** Fritsch & Carlson, "Monotone Piecewise Cubic Interpolation", SIAM J. Numer. Anal. 1980

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**For:** Master Thesis - GPRO Reward Function Analysis

