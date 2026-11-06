"""Build TWS_Solution_v11.ipynb from tws_v11.py's CELL markers, with section markdown."""
import re, json, pathlib
SRC = pathlib.Path("d:/Water_4_9_26/Water_Competition/tws_v11.py")
DST = pathlib.Path("d:/Water_4_9_26/Water_Competition/TWS_Solution_v11.ipynb")
src = SRC.read_text(encoding="utf-8")
parts = re.split(r"^# ={21} CELL (\d+) ={21}\n", src, flags=re.M)
lead, blocks = parts[0], [(int(parts[i]), parts[i+1]) for i in range(1, len(parts), 2)]

INTRO = """# A Step Ahead of Drought - Global Total Water Storage Forecasting (v11)

One-to-seven-month forecasts of GRACE Total Water Storage anomalies on a global 1 degree land
grid. Fully causal: nothing dated later than month *t* is ever read.

## Measured history

| run | public LB | what it added |
|---|---|---|
| v6 | 0.658600 | no EOF stage |
| v7 | 0.653156 | causal EOF / spatial basis |
| v8 | 0.657123 | gap-faithful validation + per-horizon calibration -- **worse** |
| v9 | **0.651802** | EOF delta bug fix, recency weighting |
| v10 | 0.653580 | good features, but a seed was cut and features pruned -- **worse** |

Two lessons are paid for and unambiguous. **Only information moves this score** -- EOF bought
-0.0054, v8's purely structural changes bought exactly 0.000, and its post-hoc calibration cost
+0.0040. And **cheap configuration savings are not cheap** -- v10's features were sound (every
base learner improved on the shared holdout) but cutting one seed and pruning on holdout-derived
SHAP cost more than the features gained.

## What was measured before v11 was built

Every external source on disk was run through one LightGBM harness at the test set's own
horizon mix, identical hyper-parameters in every arm (`diagnostics/07_value_of_information.py`):

| arm | added to the causal baseline (0.6374) | RMSE | delta |
|---|---|---|---|
| SEAS5 forecast of the target month | ECMWF seasonal, forecastMonth=2 | 0.6391 | **+0.0017** |
| provided SPEI / soil moisture from the t+1 row | | 0.6370 | -0.0004 |
| **GDO SPI-3 / SPI-6 / fAPAR anomaly** | Copernicus Global Drought Observatory | **0.6342** | **-0.0032** |
| *perfect* target-month ERA5-Land | the ceiling on any forecast product | 0.6382 | +0.0008 |

A one-month seasonal forecast explains 4% of precipitation-anomaly variance and adds noise. Even
the *true* weather of the month being predicted adds nothing -- the residual error is not in the
weather. The only channel that pays is an independent **observation** at time *t*.
A second study (`diagnostics/10_where_is_the_gap.py`) handed the model the true masked TWS at
*t*: RMSE fell only 0.6374 -> 0.6019. Everything below that is the one-step innovation of GRACE
plus its own noise, and is not reachable by any causal model.

## What v11 does

**M1 - v9's configuration, untouched.** `SEEDS=(1,2,3)`, no SHAP pruning, calibration and band
de-bias forced off, recency weighting tau=120. Nothing forecast-based is added.

**M2 - Global Drought Observatory observations, multi-source.** SPI-3 and SPI-6 from three
independent precipitation analyses -- GPCC (gauge), CPC (a different gauge network), JRA-3Q (a
reanalysis that is not ERA5). Their cross-source correlation is only 0.53-0.64, so the mean is a
materially better estimate of the field than any member, and their **spread** marks where
precipitation itself is poorly observed. Plus the MODIS **fAPAR anomaly**: satellite-observed
vegetation stress, a channel that was empty because ERA5-Land's LAI is a fixed climatology
(`x_lai` ~ 0 SHAP). Each panel enters at the cell, smoothed at sigma=1.5 and sigma=4.0, as its
change across the blind gap and as its mean over the gap. `g_spi3 - SPEI_03` separates
precipitation supply from evaporative demand.

The provided SPEI columns exist only for months present in `Test.csv`, so across the 3-18 month
voids they are NaN at prediction time. GDO is complete for every month; it fills exactly that gap.

**Not carried over from the v10/v11 plans, on evidence:** SEAS5 (measured worse), climate
indices (unmeasured, source file not on disk), per-cell residual correction and a tropics
specialist (the same post-hoc-correction family that cost v8 +0.0040 through era transfer).

## Compliance

Every feature is dated at month *t* or earlier. GDO SPI for month *t* is published in the first
days of *t+1* -- the same footing as the organisers' own `SPEI_t`. No GDO product contains GRACE
information. All external data is Copernicus (CDS, GDO); retrieval and regridding scripts are
`gdo_download.py`, `gdo_prepare.py`, `gdo_prepare_spi.py`, `era5_extra_download.py`.

## Discipline

HOLD is reported, never consulted. Every tuning decision -- stack weights, smoothing, specialist
blend weight -- is made on CAL. Consulting HOLD is precisely what cost v8 its 0.0040.

## Honest expectation

**0.647-0.650 on the public leaderboard.** The GDO gain was measured at -0.0032 in a single-model
harness; the multi-source mean is unmeasured on top of it. The error-floor analysis puts the
reachable band at 0.643-0.648 with everything done well; this is one step into it.
"""

MD = {
3:  "## 1. Load and build the dense panel\n\n`target(t)` is TWS at *t+1*, so it fills the panel one month forward wherever `TWS_t` is absent.",
5:  "## 2. Recover the hidden structure\n\n`TWS_t` is masked for 66.5% of test rows, so the effective horizon runs 1-7 months and h=1 is only a third of them. The training sampler draws its (horizon, availability) pairs from the real test blocks.",
7:  "## 3. Two-timescale memory\n\nA single AR(1) fits the global autocorrelation with SSE 0.099. Splitting into slow and fast components plus white noise fits ~48x better, and that decomposition is what the Kalman filter carries per cell.",
9:  "## 3b. The error floor\n\nFloor **C** is observation noise alone - unbeatable. Floor **B** assumes the state is perfectly nowcast at *t* but the forecaster is blind to the predicted month's weather.\n\nThe v7 -> v8 experiment is direct evidence that floor B binds in practice: four structural changes that did not add information moved the score by zero.",
11: "## 4. Expanding causal statistics, and the spatial basis\n\nEvery statistic is expanding: the value at month *m* uses months < *m* only. The EOF basis follows the same rule, refit on a checkpoint grid.",
13: "## 4b. ERA5-Land (Copernicus CDS)\n\nExplicitly permitted by the rules. The latest month any ERA5 feature reads is t = target-1, so the block stays causal. ERA5-Land does not assimilate GRACE, so no TWS information enters.",
14: "## 4c. Global Drought Observatory (Copernicus EMS) -- new in v11\n\nSPI-3 / SPI-6 from three independent precipitation analyses (GPCC, CPC, JRA-3Q) and the MODIS fAPAR anomaly. Observations dated at month *t* or earlier; nothing forecast. Caches are built from the raw GDO NetCDF files by `gdo_prepare.py` and `gdo_prepare_spi.py`.",
15: "## 5. The Kalman filter and the causal context\n\nAll per-cell statistics are expanding. **(v9-A)** `EOF_PREV` records the last month the spatial basis actually reconstructed, so the EOF delta features are well-defined at prediction time.",
17: "## 6. Features\n\nNo latitude, no longitude, no per-cell mean: regime descriptors are physical, so they transfer to cells never seen in training. **(v11)** the `g_*` block reads each GDO panel at *t*, at two spatial widths, as its change since the anchor and as its mean across the blind gap.",
19: "## 7. Training samples - the test's own horizon mix",
21: "## 8. The baseline ladder and a three-family ensemble\n\n**(v9-B)** Recency weights are applied to every booster family here. Stack weights are fitted on CAL by non-negative least squares, never on HOLD.",
23: "## 8b. What the target-month forcing is worth\n\nThe rules say only information available at or before *t* may be used, so `USE_TARGET_MONTH_FORCING` is `False`. This section prices that compliance choice.",
25: "## 9. Specialists, and the two corrections that are now switched off\n\n**(v9-C, v9-D)** Per-horizon calibration and per-band de-bias are forced off. Calibration was measured at +0.0040 on the leaderboard; band de-bias was selected by comparing on HOLD, which is not trustworthy.",
27: "## 10. Robustness across space and time",
29: "## 11. Transparency - SHAP",
31: "## 12. Bias audit",
33: "## 13. Final refit and submission\n\nRefit on the full record with the same recency weights, then predict the real test rows.",
35: "## 14. What ERA5-Land contributed",
37: "## 15. Run summary\n\nIncludes the share of SHAP attribution carried by the v11 GDO block.",
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
