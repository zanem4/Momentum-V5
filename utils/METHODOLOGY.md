# Research methodology & analysis pipeline

This document captures the intended workflow for **Momentum V5**: calibration discipline, how joint states are defined, and how **in-sample (IS)** relates to **out-of-sample (OOS)** evaluation. Revise as you rebuild the backtester.

---

## 1. Scope and goals

**Scientific intent (hypothesis shape)**  
Given measurable **liquidity** and **price-action state** (e.g. spread-related quantity and normalized acceleration), an **N-bar ordered candle** setup may produce trade outcomes that are **usable** after costs—subject to honest time separation between tuning and evaluation.

**Engineering intent**  
Keep a **repeatable pipeline**: same conventions for calibration artifacts and evaluation runs; swap **data quality** (e.g. spread from quotes) and **execution realism** without ad hoc rewrites.

---

## 2. Quantile edges: calibration key

Joint bins combine **spread** and **acceleration** ranks. **`norm_accel` is computed using ATR**, so each **`atr_length`** defines a **different** acceleration scale.

**Freeze edges at this granularity:**

\[
\text{edges} = f(\texttt{timeframe},\ L,\ \texttt{atr\_length},\ \texttt{direction})
\]

For each tuple \((\texttt{timeframe}, L, \texttt{atr\_length}, \texttt{direction})\), calibration produces (at minimum):

- **`spread_delta`** marginal quantile edges (long vs short separately if your pipeline keeps sides distinct).
- **`norm_accel`** marginal quantile edges **for that `atr_length` only**.

**Rules**

- Do **not** pool **`norm_accel` quantiles across different `atr_length`** unless you replace the metric with one that does not depend on ATR (different hypothesis).
- **Robustness across ATR** is assessed by repeating the same **rule shape** (e.g. allowed bin rectangle) **per `atr_length`**, then summarizing passes/fails—not by merging accel bins across ATR variants.

On **OOS**, label each trade with the **frozen edge dictionary** matching its simulated \((\texttt{timeframe}, L, \texttt{atr\_length}, \texttt{direction})\).

---

## 3. Joint bins: definition and semantics

Joint bins are built from **marginal quantiles** on `spread_delta` and `norm_accel`, then combined into an \(N \times N\) grid (`n_quantiles` in config). Encoding:

`bin_id = spread_bin * n_quantiles + norm_accel_bin`

**Interpretation**

- Bin indices are **ranks relative to the calibration distribution** used for that \(f(\cdot)\) key—not universal absolute spread/accel unless edges are fixed numeric thresholds elsewhere.
- If quantiles were fit **inside each rolling slice** during exploration, cross-window plots compare **within-slice ranks**, not necessarily fixed economic regimes—prefer **IS-frozen edges** per §2 when you want comparable labels across time.

Continuous **`spread_delta`** and **`norm_accel`** live at signal time in **`strategy/calculate_metrics.py`** until you extend exports.

---

## 4. Joint grid resolution: \(k\), rolling windows \(W\), and \(E_{\min}\)

Marginal quantiles (§2–3) give **approximately equal count along each axis**; the **\(k \times k\)** product grid does **not** equalize mass in **every 2D cell** when the two metrics co-move—average-count heuristics are **not** guarantees for sparse corners. The design goal here is **interpretable joint states** and **conditional return distributions** across those states, not a single equal-mass 2D partition.

### 4.1 Choosing \(k\) and \(W\) from sample size

Use a **prespecified** mapping from setup count **`n`** (e.g. `len(setup_indices)` for a calibration key) to:

- **`k`** — marginal bins per axis → **`k²`** joint cells.
- **`W`** — number of **rolling / verification windows** inside **IS** for stability views (time robustness vs spatial resolution).

A practical anchor: target a **ballpark mean count per joint cell per window** of order **`E_{\min}`** under a **uniform fantasy** over cells and windows:

\[
k \approx \left\lfloor \sqrt{\frac{n}{W \cdot E_{\min}}} \right\rfloor
\]

Clamp with **`k_min`**, **`k_max`** so sparse runs stay coarse (e.g. **`k = 2`**) and dense runs do not over-resolve (estimation noise, multiple comparisons). Larger **`W`** demands smaller **`k`** for fixed **`n`** and **`E_{\min}`**—a deliberate trade: **more temporal slices vs finer binning**. Different **timeframes** may legitimately yield different **`W`** as long as **all windows stay inside IS** and **`(W, k)`** is chosen by **published rules** (e.g. tied to **`n`**) **before** outcome-chasing, not tuned per TF after inspecting performance.

After fixing **`k`**, **tabulate observed counts** per **`bin_id`** (and per window if applicable). If too many cells fall below **`E_{\min}`**, reduce **`k`**, reduce **`W`**, merge tails under a prespecified rule, or extend calibration span—do not rely on the formula alone.

Implementation sketch: **`strategy/quantile.py`** (and parameters as needed).

### 4.2 What **`E_{\min}`** is (and is not)

Formally, detecting a modest edge versus noise often scales like **\(n \propto \sigma^2 / \delta^2\)** (dispersion vs smallest economically relevant shift \(\delta\), up to constants involving \(z\)-levels). That back-of-envelope suggests **how large **`n`** per decision unit might need to be** under idealized single-cell testing.

In practice **`E_{\min}`** is set in the **20–30** range as a **chosen tolerance for estimation noise**, not as that theorem evaluated cell-by-cell, because:

- Decisions rely on **cross-bin structure** (marginals, corners vs interior, qualitative patterns across liquidity × accel).
- Decisions rely on **cross-time structure** (**`W`** windows, stability narratives)—not a single heatmap tile.
- Joint bins are **unequal mass**; average \(\approx E_{\min}\) **does not** imply every cell clears **`E_{\min}`**.

Treat **`E_{\min}`** as a **documented robustness knob**, optionally motivated by **\((\sigma/\delta)^2\)** intuition, then **downshifted** for aggregated evidence—align with **`MIN_BIN_SAMPLE_COUNT`** (§6) where summaries mask noisy moments.

---

## 5. Stop / target simulation

**TBD.** Calibration may use a grid for exploration and end on a **single** `(stop, target)` or band for live-like replay; resolve separately from liquidity × accel state labeling.

---

## 6. Sample counts and masked statistics

In **`strategy/risk_return_distributions.py`**, summaries below **`MIN_BIN_SAMPLE_COUNT`** (default 30) keep **`n_count`** but can zero out finite mean/median/variance for sparse bins.

Aggregations that weight only finite moments may show **blank panels** while **`pnl_n_count > 0`** still exists below threshold—always inspect **raw counts** where available.

---

## 7. In-sample / out-of-sample protocol (intended)

### 7.1 Split

| Phase | Period | Role |
|-------|--------|------|
| **IS** | e.g. 2023–2024 | Calibration: structural grid, define gates, **freeze edges** per §2. |
| **OOS** | e.g. 2025 | **Single** frozen evaluation (no retuning on OOS). |

Optional: walk-forward **within IS** before trusting the single OOS slice.

### 7.2 Global quantiles within IS

Typical sequence:

1. On **IS only**, estimate quantile edges **separately for each** \(f(\texttt{timeframe}, L, \texttt{atr\_length}, \texttt{direction})\) (or on an explicit **calibration subset** of IS for stricter causal labeling early in IS).
2. **Freeze** all edge tables **before** any OOS data.
3. Label trades on remaining IS verification slices and on OOS using **only** those frozen edges.

**Caveat inside IS:** fitting edges on **full** IS then labeling trades from **early** IS uses **later-IS** information in the label—fine for OOS hygiene; stricter discipline uses calibration vs verification subsets inside IS.

### 7.3 Continuous simulation vs rolling windows

Prefer **one continuous simulation** with trade timestamps, then **aggregate into rolling calendar windows** for stability summaries. Rolling aggregation does not fix bin semantics unless edges follow §2.

### 7.4 Execution realism IS vs OOS

OOS may use stricter spread, commissions, and paths—often conservative. For apples-to-apples headlines, optionally **replay IS** with the **same** execution module used for OOS.

---

## 8. Hypothesis framing (make it falsifiable)

| Informal phrase | Operational example |
|-----------------|----------------------|
| Optimal numeric band | Trade iff \(a_1 < X < a_2\) under **frozen** edges per §2; bounds from calibration IS only. |
| Favorable liquidity | e.g. `spread_delta` bin/threshold under frozen spread edges for that \(f(\cdot)\). |
| Exploitable | e.g. median(return) **>** costs + buffer vs prespecified benchmark. |
| Stable through time | Prespecified gates per verification window or fraction of windows ≥ \(p\) with \(n \geq N_{\min}\). |

**Correlation ≠ causation** — predictive stability may suffice for deployment; causality helps monitoring.

---

## 9. Strict tests (minimal checklist)

1. Freeze \((\texttt{timeframe}, L, \texttt{atr\_length}, F, \texttt{direction})\) or explicit combo rule where possible; resolve stop/target per §5.
2. Freeze liquidity/accel gates using **edges** per §2 from calibration IS only.
3. Per rolling or verification slice: one scalar (e.g. median return) + **`n`** on gated trades.
4. Compare to prespecified hurdles **before** trusting OOS.

---

## 10. Multiple comparisons

Many bins × windows × variants ⇒ **implicit multiple testing**. Prefer one **primary** metric, verification splits inside IS, and **scalar stability summaries** (not only heatmaps).

---

## 11. Revision log

| Date | Notes |
|------|--------|
| (initial) | First consolidated methodology. |
| Revision | Restart-oriented doc; **edges = f(timeframe, L, atr_length, direction)**; dropped legacy repo artifact inventory; stop/target **TBD**. |
| Revision | §4 **Joint grid resolution**: dynamic **`k`**, **`W`**, **`E_{\min}`** rationale; **`E_{\min}`** as tolerance vs formal \(n\) scaling; §§5–11 renumbered (was §§4–10). |

_Add rows when calibration rules or codepaths change._
