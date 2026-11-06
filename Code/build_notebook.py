"""Build TWS_Solution_v9.ipynb from tws_v9.py's CELL markers, with section markdown."""
import re, json, pathlib
SRC = pathlib.Path("d:/Water_4_9_26/Water_Competition/tws_v9.py")
DST = pathlib.Path("d:/Water_4_9_26/Water_Competition/TWS_Solution_v9.ipynb")
src = SRC.read_text(encoding="utf-8")
parts = re.split(r"^# ={21} CELL (\d+) ={21}\n", src, flags=re.M)
lead, blocks = parts[0], [(int(parts[i]), parts[i+1]) for i in range(1, len(parts), 2)]

INTRO = """# A Step Ahead of Drought - Global Total Water Storage Forecasting (v9)

One-to-seven-month forecasts of GRACE Total Water Storage anomalies on a global 1 degree land
grid. Fully causal: nothing dated later than month *t* is ever read.

## Why v9 returns to the v7 base

Three configurations have now been measured on the public leaderboard:

| run | public LB | what it added |
|---|---|---|
| v6 | 0.6586 | no EOF stage |
| **v7** | **0.653156** | causal EOF / spatial basis |
| v8 | 0.657123 | gap-faithful validation, h>=4 specialist, long-void augmentation, out-of-fold flag selection |

**v8 was worse.** The damage decomposes cleanly: its per-horizon calibration cost **+0.0044**
on its own holdout and **+0.0040** on the board. Those two agreeing to within a thousandth is
strong evidence that the calibration *was* the regression, and that the other three structural
changes together netted **zero**.

That is the central lesson of this project, and it matches the error-floor analysis in section
3b: **you cannot beat floor B by tuning, only by adding information.** The only change that has
ever improved this score is the EOF stage (-0.0054), which added spatial information.

So v9 returns to the v7 base and keeps only changes with evidence behind them:

- **A. EOF delta bug fix.** v7 differenced the EOF field against `anc-1`, but `eof_panels`
  leaves a month NaN when fewer than 200 cells are observed. Test-era anchors are 5-6 months
  apart, so `anc-1` was almost never reconstructed: all four `eof*_d` features were populated
  in training and systematically NaN at prediction time. Now differenced against the last month
  the basis actually reconstructed, with the staleness exposed as `eof_dt`.
- **B. Recency weighting.** Calibration failed because coefficients fitted on one era did not
  transfer to another. The same era gap applies to the boosters, only more gently, so training
  rows are weighted by target recency (tau = 120 months, normalised to mean 1).
- **C. Per-horizon calibration forced OFF** - measured harmful.
- **D. Per-band de-bias forced OFF** - v7 selected it by comparing on HOLD, which is not a
  trustworthy signal, and it is the same family of post-hoc additive correction that cost v8.

Deliberately **not** carried over from v8: the gap-faithful CAL/HOLD split (its tuning
transferred worse), the h>=4 specialist (CAL chose blend weight 0.0), and long-void
augmentation (net zero).

## Honest expectation

This is expected to land near **0.645-0.650**, not 0.61. Reaching the low 0.62s requires
information this pipeline does not yet have: GDO/EDO fAPAR and LISFLOOD soil moisture, ENSO
indices derived from ERA5 SST, and a basin-scale hierarchy. Those need Copernicus downloads and
are documented as the next step, not implemented here."""

MD = {
3:  "## 1. Load and build the dense panel\n\n`target(t)` is TWS at *t+1*, so it fills the panel one month forward wherever `TWS_t` is absent.",
5:  "## 2. Recover the hidden structure\n\n`TWS_t` is masked for 66.5% of test rows, so the effective horizon runs 1-7 months and h=1 is only a third of them. The training sampler draws its (horizon, availability) pairs from the real test blocks.",
7:  "## 3. Two-timescale memory\n\nA single AR(1) fits the global autocorrelation with SSE 0.099. Splitting into slow and fast components plus white noise fits ~48x better, and that decomposition is what the Kalman filter carries per cell.",
9:  "## 3b. The error floor\n\nFloor **C** is observation noise alone - unbeatable. Floor **B** assumes the state is perfectly nowcast at *t* but the forecaster is blind to the predicted month's weather.\n\nThe v7 -> v8 experiment is direct evidence that floor B binds in practice: four structural changes that did not add information moved the score by zero.",
11: "## 4. Expanding causal statistics, and the spatial basis\n\nEvery statistic is expanding: the value at month *m* uses months < *m* only. The EOF basis follows the same rule, refit on a checkpoint grid.",
13: "## 4b. ERA5-Land (Copernicus CDS)\n\nExplicitly permitted by the rules. The latest month any ERA5 feature reads is t = target-1, so the block stays causal. ERA5-Land does not assimilate GRACE, so no TWS information enters.",
15: "## 5. The Kalman filter and the causal context\n\nAll per-cell statistics are expanding. **(v9-A)** `EOF_PREV` records the last month the spatial basis actually reconstructed, so the EOF delta features are well-defined at prediction time.",
17: "## 6. Features\n\nNo latitude, no longitude, no per-cell mean: regime descriptors are physical, so they transfer to cells never seen in training.",
19: "## 7. Training samples - the test's own horizon mix",
21: "## 8. The baseline ladder and a three-family ensemble\n\n**(v9-B)** Recency weights are applied to every booster family here. Stack weights are fitted on CAL by non-negative least squares, never on HOLD.",
23: "## 8b. What the target-month forcing is worth\n\nThe rules say only information available at or before *t* may be used, so `USE_TARGET_MONTH_FORCING` is `False`. This section prices that compliance choice.",
25: "## 9. Specialists, and the two corrections that are now switched off\n\n**(v9-C, v9-D)** Per-horizon calibration and per-band de-bias are forced off. Calibration was measured at +0.0040 on the leaderboard; band de-bias was selected by comparing on HOLD, which is not trustworthy.",
27: "## 10. Robustness across space and time",
29: "## 11. Transparency - SHAP",
31: "## 12. Bias audit",
33: "## 13. Final refit and submission\n\nRefit on the full record with the same recency weights, then predict the real test rows.",
35: "## 14. What ERA5-Land contributed",
37: "## 15. Run summary",
}

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t.rstrip("\n").splitlines(keepends=True)})

md(INTRO)
code(lead + blocks[0][1].rstrip("\n"))
for n, body in blocks[1:]:
    if n in MD: md(MD[n])
    code(body.rstrip("\n"))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.10"}},
      "nbformat": 4, "nbformat_minor": 5}
DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {DST} | {len(cells)} cells ({sum(c['cell_type']=='code' for c in cells)} code)")
