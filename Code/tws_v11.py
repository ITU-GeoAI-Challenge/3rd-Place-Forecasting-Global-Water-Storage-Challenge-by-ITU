
# ===================== CELL 1 =====================
import os, sys, time, json, warnings, pickle, bisect
NTHREAD = int(os.environ.get("TWS_THREADS", "6"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(NTHREAD))
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
import lightgbm as lgb
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DATA_DIR = Path(".")                       # folder holding Train.csv / Test.csv
OUT      = Path(os.environ.get("TWS_OUT", "outputs_v11")); OUT.mkdir(exist_ok=True)
SEEDS    = (1, 2, 3)
MAXIT    = 900                              # ceiling; early stopping on CAL picks the number
N_ITER   = MAXIT                            # (kept as the name the rest of the notebook uses)
N_ITER_F = None                             # derived from the early-stopped best iteration
PER_MONTH, PER_MONTH_F = 15715, 15715       # (B) every cell, not a 70% sample
REPS,      REPS_F      = 2, 2               # (B) horizon draws per (anchor, cell)

SMOKE = bool(int(os.environ.get("TWS_SMOKE", "0")))   # 1 = fast structural test of every cell
if SMOKE:
    SEEDS, MAXIT, N_ITER, PER_MONTH, PER_MONTH_F = (1,), 60, 60, 400, 400
    REPS, REPS_F = 1, 1

# =====================================================================================
#  USE_TARGET_MONTH_FORCING  --  READ THIS BEFORE CHANGING IT
# -------------------------------------------------------------------------------------
#  True  : SPEI and soil moisture dated at the month being predicted (t+1) are used as
#          features. The values come from other rows of Test.csv, so they were supplied
#          by the organisers -- but they are dated AFTER the prediction time t, and the
#          problem statement says:
#             "For predictions at month t, only information available at or before t
#              may be used."
#          A strict reading of that rule forbids these features. Set False for a fully
#          compliant model; section 8b prices the difference on held-out data.
#  False : nothing in the pipeline is ever dated later than t. Leak-free by construction.
# =====================================================================================
USE_TARGET_MONTH_FORCING = False   # (compliance) nothing dated later than t is read

# =====================================================================================
#  USE_ERA5 -- ERA5-Land from the Copernicus Climate Data Store.
#  Explicitly permitted by the rules. The latest month ever read is t = target-1, so the
#  ERA5 block stays causal even when the flag above is False. See section 4b.
# =====================================================================================
USE_ERA5 = True

# =====================================================================================
#  EOF / spatial-basis stage.
#  The TWS panel has ~15,700 land cells but only ~50 spatial degrees of freedom: 36 EOFs
#  carry 90% of the variance, 55 carry 95%. That is not a quirk of this dataset -- GRACE
#  Level-2 is spherical harmonics truncated near degree 60 (~300 km), so the 1-degree grid
#  is an interpolation of a much lower-dimensional object.
#  Truncating the expansion is therefore denoising: signal concentrates in the leading
#  components, observation noise spreads across all of them.
#  CAUSALITY: the basis at month m is fitted on months < m only, refit every EOF_CKPT
#  months -- the same expanding-checkpoint pattern used by cell_params_causal.
# =====================================================================================
USE_EOF  = bool(int(os.environ.get("TWS_EOF", "1")))
EOF_K    = (10, 20, 30, 50)     # measured: blend gain peaks near K=30, grows as K coarsens
EOF_CKPT = 12                  # refit the basis every 12 months
EOF_MIN  = 48                  # need this many past months before a basis is trusted

# =====================================================================================
#  v9 -- what the leaderboard actually taught us
#  v7 scored 0.653156. v8 added gap-faithful validation, an h>=4 specialist, long-void
#  augmentation and out-of-fold flag selection, and scored 0.657123 -- WORSE by 0.0040.
#  The damage decomposes cleanly: v8's per-horizon calibration cost 0.0044 on its own
#  holdout and 0.0040 on the board, so the other three changes together netted zero.
#  v9 therefore returns to the v7 base and keeps only what is defensible:
#    - the EOF delta bug fix (v7 differenced against anc-1, which is unobserved at test time)
#    - recency weighting, which attacks the era-transfer failure calibration exposed
#    - calibration and band de-bias forced OFF, both measured or suspected harmful
# =====================================================================================
RECENCY_TAU = float(os.environ.get("TWS_TAU", "120"))   # months; 0 disables the weighting
FORCE_NO_CAL = bool(int(os.environ.get("TWS_NOCAL", "1")))

# =====================================================================================
#  v11 -- v9 plus the one external channel that measured positive
#  A value-of-information study (diagnostics/07) ran every source now on disk through the
#  same LightGBM harness at the test's own horizon mix:
#      + SEAS5 forecast of the target month        +0.0017  (worse: r^2 = 0.04 skill)
#      + provided SPEI/SM read from the t+1 row    -0.0004  (noise)
#      + GDO SPI-3 / SPI-6 / fAPAR anomaly         -0.0032  (the only gain)
#      + PERFECT target-month ERA5-Land             +0.0008  (the ceiling on ANY forecast)
#  So v11 adds the Global Drought Observatory observations and nothing forecast-based.
#    - SPI-3 / SPI-6 from three independent precipitation analyses: GPCC (gauge), CPC
#      (a different gauge network) and JRA-3Q (a reanalysis that is not ERA5). They agree
#      only at r = 0.53-0.64, so the three-source mean is a better estimate of the field
#      than any member and their spread marks where precipitation is poorly observed.
#    - fAPAR anomaly (MODIS): satellite-OBSERVED vegetation stress. ERA5-Land's LAI is a
#      fixed climatology (x_lai ~ 0 SHAP), so this channel was empty.
#  Everything is dated at month t or earlier. SPI for month t is published by GDO in the
#  first days of t+1 -- the same footing as the organisers' own SPEI_t.
#  The provided SPEI columns exist ONLY for months present in Test.csv, so across the 3-18
#  month voids the sp*_l1..l6 features are NaN at prediction time. GDO is complete for
#  every month, so it fills exactly the gap the provided covariates leave.
#  Model, evaluation, seeds (1,2,3) and feature set are otherwise v9's, untouched.
# =====================================================================================
USE_GDO = bool(int(os.environ.get("TWS_GDO", "1")))
GDO_DIR = Path("gdo")

FAST = bool(int(os.environ.get("TWS_FAST", "0")))
USE_EXTRA = not FAST          # XGBoost + CatBoost: worth ~+0.0015, cost ~1.3 h
PRICE_FT  = bool(int(os.environ.get("TWS_PRICE_FT", "0")))   # diagnostic only, ~1h, no gain
if FAST:
    SEEDS = (1,)              # 1 model instead of 3 averaged  -> 3x faster
    REPS = REPS_F = 1         # one horizon draw per cell      -> 2x fewer rows
if os.environ.get("TWS_SEEDS"):
    SEEDS = tuple(int(x) for x in os.environ["TWS_SEEDS"].split(","))
if os.environ.get("TWS_REPS"):
    REPS = REPS_F = int(os.environ["TWS_REPS"])
print(f"FAST={FAST} | SEEDS={SEEDS} | REPS={REPS} | extra_models={USE_EXTRA}")

rng = np.random.default_rng(7)
T0  = time.time()
RESULTS = {}
print("lightgbm", lgb.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)
print(f"USE_TARGET_MONTH_FORCING={USE_TARGET_MONTH_FORCING} | USE_ERA5={USE_ERA5} | USE_GDO={USE_GDO} | SMOKE={SMOKE}")


# ===================== CELL 3 =====================
t = time.time()
COLS = ["time","lat","lon","TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
train = pd.read_csv(DATA_DIR/"Train.csv", usecols=COLS+["target"], parse_dates=["time"])
test  = pd.read_csv(DATA_DIR/"Test.csv",  usecols=COLS+["ID"],     parse_dates=["time"])
print(f"train {len(train):,} rows | test {len(test):,} rows | {time.time()-t:.0f}s")

for d in (train, test):
    d["m"] = d.time.dt.year*12 + d.time.dt.month

cells = (pd.concat([train[["lat","lon"]], test[["lat","lon"]]])
           .drop_duplicates().sort_values(["lat","lon"]).reset_index(drop=True))
cells["c"] = np.arange(len(cells))
lat, lon = cells.lat.to_numpy(np.float64), cells.lon.to_numpy(np.float64)
_NLAT, _NLON, _LAT0, _LON0 = 140, 360, -55.5, -179.5
CELLID = np.full((_NLAT, _NLON), -1, np.int64)
CELLID[np.round(lat-_LAT0).astype(int), np.round(lon-_LON0).astype(int)] = cells.c.to_numpy()
def cidx(dlat, dlon):
    """(lat,lon) -> cell index through the integer grid: exact, no float dict keys."""
    i = CELLID[np.round(np.asarray(dlat, np.float64)-_LAT0).astype(int),
               np.round(np.asarray(dlon, np.float64)-_LON0).astype(int)]
    assert (i >= 0).all(), "cell outside the challenge grid"
    return i

m_lo = int(min(train.m.min(), test.m.min()))
m_hi = int(max(train.m.max(), test.m.max())) + 2
nM, nC = m_hi - m_lo + 1, len(cells)
FC = ["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
SH = {"SPEI_01_t":"sp1","SPEI_03_t":"sp3","SPEI_06_t":"sp6","SPEI_12_t":"sp12","SOIL_MOISTURE_t":"sm"}

P = {k: np.full((nM,nC), np.nan, np.float32) for k in ["TWS_t"]+FC}
for d in (train, test):
    ci = cidx(d.lat.to_numpy(), d.lon.to_numpy())
    mi_ = d.m.to_numpy() - m_lo
    for k in ["TWS_t"]+FC:
        P[k][mi_, ci] = d[k].to_numpy(np.float32)

# target(t) is TWS at t+1 -> it fills the TWS panel one month forward where not already present
ci = cidx(train.lat.to_numpy(), train.lon.to_numpy()); mi_ = train.m.to_numpy() - m_lo
tg = train.target.to_numpy(np.float32); ok = np.isfinite(tg)
cur = P["TWS_t"][mi_[ok]+1, ci[ok]]; fill = ~np.isfinite(cur)
P["TWS_t"][(mi_[ok]+1)[fill], ci[ok][fill]] = tg[ok][fill]

TWS = P["TWS_t"]; mon = (np.arange(nM) + m_lo) % 12
def midx(y, m): return int(y*12 + m - m_lo)
print(f"panel {nM} months x {nC} cells | TWS coverage {np.isfinite(TWS).mean():.3f}")

# ===================== CELL 5 =====================
obs_m = np.isfinite(TWS).mean(axis=1) > 0.5
has_f = np.isfinite(P["SPEI_03_t"]).mean(axis=1) > 0.5
tmon  = sorted(test.m.unique())

print("test month   rows   TWS obs   horizon   forcing signature (target-1..-6)")
BLOCKS = []
for mm in tmon:
    m = mm - m_lo
    n = int((test.m == mm).sum())
    a = m
    while a >= 0 and not obs_m[a]: a -= 1
    h   = m + 1 - a
    tgt = m + 1
    sig = tuple(int(has_f[tgt-k]) if 0 <= tgt-k < nM else 0 for k in range(1,7))
    ftv = int(has_f[tgt]) if 0 <= tgt < nM else 0      # forcing AT the month being predicted
    BLOCKS.append((h, sig + (ftv,), n))
    y, k = divmod(mm-1, 12)
    print(f"  {y}-{k+1:02d}   {n:6,}   {100*np.isfinite(TWS[m]).mean():5.1f}%   h={h}   {sig}  ft={ftv}")

TEST_W = {}
for h,_,n in BLOCKS: TEST_W[h] = TEST_W.get(h,0) + n
tot = sum(TEST_W.values())
print("\nhorizon mix:", {h: f"{100*TEST_W[h]/tot:.1f}%" for h in sorted(TEST_W)})
print(f"h=1 is only {100*TEST_W[1]/tot:.1f}% of rows -- this is NOT a one-month-ahead problem")

FT_SHARE = float(sum(n for _,sg,n in BLOCKS if sg[6])/sum(n for _,_,n in BLOCKS))
print(f"rows whose TARGET month has forcing: {100*FT_SHARE:.1f}%  <-- training must match this")
_w   = np.array([b[2] for b in BLOCKS], float); PROB = _w/_w.sum()
HOR  = np.array([b[0] for b in BLOCKS]); SIGA = np.array([b[1] for b in BLOCKS], np.float32)
def draw(n, r):
    i = r.choice(len(BLOCKS), size=n, p=PROB)
    return HOR[i].astype(np.float32), SIGA[i]

# ===================== CELL 7 =====================
from scipy.optimize import minimize
clim_full = np.stack([np.nanmean(np.where(mon[:,None]==k, TWS, np.nan), axis=0) for k in range(12)])
DEVf = TWS - clim_full[mon]; DEVf = DEVf - np.nanmean(DEVf, axis=0)

L = 14
rho = np.array([(lambda a,b,o: np.corrcoef(a[o],b[o])[0,1])(
        DEVf[:-k], DEVf[k:], np.isfinite(DEVf[:-k])&np.isfinite(DEVf[k:])) for k in range(1,L+1)])
kk  = np.arange(1, L+1)
lamA = rho[0]**2/rho[1]; phiA = rho[1]/rho[0]
sseA = float(np.sum((lamA*phiA**kk - rho)**2))

def obj(z):
    a,b,p,q = z
    return np.sum((a*p**kk + b*q**kk - rho)**2)
best = None
for p0 in (0.93,0.96,0.98):
    for q0 in (0.3,0.5,0.7):
        r = minimize(obj, [0.5,0.4,p0,q0], method="SLSQP",
                     bounds=[(0,1),(0,1),(0.85,0.999),(0.0,0.85)],
                     constraints=[{"type":"ineq","fun": lambda z: 1-z[0]-z[1]}])
        if best is None or r.fun < best.fun: best = r
a,b,p,q = best.x; noise = 1-a-b
print(f"AR(1)+white   SSE {sseA:.5f}   phi={phiA:.4f}  noise={100*(1-lamA):.1f}%")
print(f"2-comp+white  SSE {best.fun:.5f}   slow phi={p:.4f} (w={a:.3f}, e-fold {-1/np.log(p):.1f} mo)"
      f"  fast phi={q:.4f} (w={b:.3f})  noise={100*noise:.1f}%")
print(f"   -> the two-component model fits {sseA/best.fun:.0f}x better\n")
print("  lag       ", "  ".join(f"{k:5d}" for k in range(1,9)))
print("  empirical ", "  ".join(f"{v:5.3f}" for v in rho[:8]))
print("  AR(1)     ", "  ".join(f"{v:5.3f}" for v in (lamA*phiA**kk)[:8]))
print("  2-comp    ", "  ".join(f"{v:5.3f}" for v in (a*p**kk+b*q**kk)[:8]))
RESULTS.update(sse_ar1=sseA, sse_2comp=float(best.fun), phi_slow=float(p), phi_fast=float(q),
               noise_share=float(noise))

# ===================== CELL 9 =====================
def era_var(a, b):
    return float(np.nanvar(DEVf[a:b]))
print("anomaly variance by era")
for nm, a, b in [("2002-2010", 0, midx(2011,1)), ("2011-2014.6", midx(2011,1), midx(2014,7)),
                 ("HOLD 2014.7+", midx(2014,7), nM), ("pooled", 0, nM)]:
    print(f"   {nm:<14} {era_var(a,b):.4f}")

obs_test = [midx(y,m) for y,m in [(2015,9),(2016,1),(2016,6),(2016,12),(2018,7),(2018,11)]]
V_test = float(np.nanvar(DEVf[obs_test]))
print(f"\n   TEST era (6 observed months)   {V_test:.4f}   <-- floors must be scaled to THIS")

W = TEST_W; WT = sum(W.values())
print(f"\n{'parameter set':<34}{'floor C':>10}{'floor B':>10}{'floor A':>10}")
floors = {}
for nm, noi, ph in [("pooled  (noise 13.4%, phi .877)", 1-lamA, phiA),
                    ("late era (noise 23.3%, phi .892)", 0.233, 0.892)]:
    sig = 1-noi
    fC = np.sqrt(V_test*noi)
    fB = np.sqrt(V_test*noi + V_test*sig*(1-ph**2))
    fA = np.sqrt(sum(W[h]/WT*(V_test*noi + V_test*sig*(1-ph**(2*h))) for h in sorted(W)))
    floors[nm] = (fC, fB, fA)
    print(f"{nm:<34}{fC:>10.4f}{fB:>10.4f}{fA:>10.4f}")
print("""
  C  observation noise alone            unattainable by any forecast
  B  state perfectly nowcast at time t  the practical limit
  A  state known only at the anchor     blind between anchor and now

  Floor B assumes the forecaster is BLIND to weather during the month being predicted.
  Measured (diagnostics/07, /10): handing the model the TRUE target-month ERA5-Land
  weather moves the score by +0.0008 -- nothing. The provided SPEI/SM columns read from
  the t+1 row are worth -0.0004, also nothing (they track month t-2, not t+1). Floor B
  is therefore the practical limit even with perfect weather, and the residual is the
  one-step innovation of GRACE itself plus its noise. USE_TARGET_MONTH_FORCING stays
  False on compliance grounds; it costs nothing.
  So: a score below floor B is not evidence of skill -- it is evidence that information
  dated after the prediction time entered the model.""")
RESULTS.update(V_test=V_test, floor_B=[floors[k][1] for k in floors],
               floor_A=[floors[k][2] for k in floors])

# ===================== CELL 11 =====================
NLAT, NLON, LAT0, LON0 = 140, 360, -55.5, -179.5
ilat = np.round(lat - LAT0).astype(int); ilon = np.round(lon - LON0).astype(int)

def nan_gauss_panel(A, sigma):
    """Spatial Gaussian smooth over the lat/lon grid, NaN-aware, longitude wrapped."""
    out = np.full_like(A, np.nan, dtype=np.float32)
    gr  = np.zeros((NLAT,NLON), np.float32); msk = np.zeros((NLAT,NLON), np.float32)
    for m in range(A.shape[0]):
        v = A[m]
        if not np.isfinite(v).any(): continue
        gr[:] = 0.0; msk[:] = 0.0
        f = np.isfinite(v)
        gr[ilat[f], ilon[f]] = v[f]; msk[ilat[f], ilon[f]] = 1.0
        num = gaussian_filter(gr,  sigma, mode=("nearest","wrap"))
        den = gaussian_filter(msk, sigma, mode=("nearest","wrap"))
        out[m] = np.where(den[ilat,ilon] > 1e-6, (num/den)[ilat,ilon], np.nan)
    return out

def grid_ops(A):
    """(D) N-S and E-W gradient and the Laplacian of a [nM,nC] panel.
       Directional structure the isotropic Gaussian smooths cannot express."""
    gN  = np.full_like(A, np.nan, dtype=np.float32)
    gE  = np.full_like(A, np.nan, dtype=np.float32)
    lap = np.full_like(A, np.nan, dtype=np.float32)
    G   = np.full((NLAT+2, NLON), np.nan, np.float32)
    for m in range(A.shape[0]):
        G[:] = np.nan
        G[ilat+1, ilon] = A[m]
        up = G[ilat+2, ilon]; dn = G[ilat, ilon]
        ea = G[ilat+1, (ilon+1) % NLON]; we = G[ilat+1, (ilon-1) % NLON]
        gN[m]  = up - dn
        gE[m]  = ea - we
        lap[m] = A[m] - np.nanmean(np.stack([up, dn, ea, we]), axis=0)
    return gN, gE, lap

def gather(A, m, c):
    o = np.full(len(m), np.nan, np.float32); ok = (m >= 0) & (m < A.shape[0])
    o[ok] = A[m[ok], c[ok]]; return o

def eof_panels(DEV, Ks=EOF_K, ckpt=EOF_CKPT, min_m=EOF_MIN):
    """Causal EOF reconstruction of the anomaly panel.

    For each month m the basis comes from months < m only (refit every `ckpt` months).
    The month is then projected onto the leading K vectors by least squares over the
    cells observed THAT month, and reconstructed everywhere. Returns one [nM,nC] panel
    per truncation K. Cells missing that month get the reconstruction too -- that is
    the point: the basis fills them from the global pattern."""
    nM_, nC_ = DEV.shape
    out = {k: np.full((nM_, nC_), np.nan, np.float32) for k in Ks}
    Kmax = max(Ks); V = None; nfit = 0
    for m in range(nM_):
        if m >= min_m and (V is None or m % ckpt == 0):
            Wp = np.nan_to_num(DEV[:m], nan=0.0)
            if Wp.shape[0] >= Kmax:
                try:
                    _, _, Vt = np.linalg.svd(Wp, full_matrices=False)
                    V = Vt[:Kmax]; nfit += 1
                except np.linalg.LinAlgError:
                    pass
        if V is None: continue
        v = DEV[m]; ok = np.isfinite(v)
        if ok.sum() < 200: continue
        vo = v[ok]
        for K in Ks:
            B = V[:K][:, ok]                       # (K, nObs)
            try:
                a, *_ = np.linalg.lstsq(B.T, vo, rcond=None)
            except np.linalg.LinAlgError:
                continue
            out[K][m] = (V[:K].T @ a).astype(np.float32)
    print(f"  EOF: {nfit} causal basis refits | K={Ks} | first usable month {min_m}")
    return out

def cell_params_causal(DEV, ckpt=12, min_m=24, ps0=0.9165, no0=0.1667):
    """(E) Expanding version: the (phi, noise) pair used at month m is estimated from
       months < m only, refreshed on a checkpoint grid so the cost stays linear.
       Before min_m there is too little history, so the documented global fit is used."""
    nMl, n = DEV.shape
    PS = np.full((nMl, n), ps0); NO = np.full((nMl, n), no0)
    ps_c = np.full(n, ps0); no_c = np.full(n, no0); last = -10**9
    for m in range(nMl):
        if m >= min_m and (m - last) >= ckpt:
            ps_c, no_c = cell_params(DEV[:m]); last = m
        PS[m] = ps_c; NO[m] = no_c
    return PS, NO

def cell_params(DEV, shrink=0.55):
    """Per-cell memory and noise from each cell's own autocorrelation, shrunk toward the
       global fit because ~140 monthly samples give noisy estimates.
       phi = rho2/rho1 ;  lambda = rho1^2/rho2  (signal share)."""
    def ac(k):
        x, y = DEV[:-k], DEV[k:]; ok = np.isfinite(x)&np.isfinite(y)
        c  = ok.sum(axis=0)
        mx = np.where(ok,x,0).sum(axis=0)/np.maximum(c,1); my = np.where(ok,y,0).sum(axis=0)/np.maximum(c,1)
        cov = np.where(ok,(x-mx)*(y-my),0).sum(axis=0)/np.maximum(c-1,1)
        vx  = np.where(ok,(x-mx)**2,0).sum(axis=0)/np.maximum(c-1,1)
        vy  = np.where(ok,(y-my)**2,0).sum(axis=0)/np.maximum(c-1,1)
        return cov/np.maximum(np.sqrt(vx*vy),1e-9), c
    r1,c1 = ac(1); r2,_ = ac(2)
    phi = np.clip(r2/np.where(np.abs(r1)>1e-6, r1, np.nan), 0.55, 0.985)
    lam = np.clip(r1**2/np.where(np.abs(r2)>1e-6, r2, np.nan), 0.45, 0.985)
    phi = np.where(np.isfinite(phi), phi, 0.9165); lam = np.where(np.isfinite(lam), lam, 0.833)
    w = np.clip(c1/120.0, 0, 1)*(1-shrink)
    return (w*phi + (1-w)*0.9165).astype(np.float64), (1 - (w*lam + (1-w)*0.833)).astype(np.float64)
def exp_anom(A):
    """Expanding, causal deseasonalisation. The monthly mean removed at month m is built
       from months < m only, so no value is ever informed by its own future."""
    D = np.full_like(A, np.nan, np.float32)
    acc = np.zeros((12,nC)); cnt = np.zeros((12,nC))
    for m in range(nM):
        k = mon[m]
        D[m] = A[m] - np.where(cnt[k] > 0, acc[k]/np.maximum(cnt[k],1), np.nan)
        v = A[m]; o = np.isfinite(v); acc[k][o] += v[o]; cnt[k][o] += 1
    return D

FIT_END_REG = midx(2013, 7)     # regime descriptors see months < 2013-07 only, never the
                                # scoring era. midx() already accounts for the panel origin.
print("helpers defined")

# ===================== CELL 13 =====================
ERA5_DIR = Path("era5")
LAYER_M  = [0.07, 0.21, 0.72, 1.89]        # ERA5-Land soil layer thicknesses, metres
CDS_DATASET = "reanalysis-era5-land-monthly-means"
ERA5_VARS = ["snow_depth_water_equivalent",
             "volumetric_soil_water_layer_1","volumetric_soil_water_layer_2",
             "volumetric_soil_water_layer_3","volumetric_soil_water_layer_4",
             "total_precipitation","total_evaporation","runoff","snowmelt"]

def era5_download(out_dir, y0=2002, y1=2019):
    """Server-side regrid keeps the archive small. Needs a free CDS account (~/.cdsapirc)."""
    import cdsapi; c = cdsapi.Client(); os.makedirs(out_dir, exist_ok=True)
    for y in range(y0, y1+1):
        dst = os.path.join(out_dir, f"era5land_{y}.nc")
        if os.path.exists(dst): continue
        c.retrieve(CDS_DATASET, {"product_type":"monthly_averaged_reanalysis",
            "variable":ERA5_VARS, "year":str(y), "month":[f"{m:02d}" for m in range(1,13)],
            "time":"00:00", "area":[84,-180,-56,180], "grid":[0.5,0.5],
            "data_format":"netcdf"}, dst)

def era5_load(nc_dir, pattern="era5land_*.nc", varlist=None):
    """-> dict of [nM, nC] panels on the challenge's own cells, by coordinate lookup."""
    import glob, xarray as xr
    files = sorted(glob.glob(str(Path(nc_dir)/pattern)))
    if not files: return None
    f0 = xr.open_dataset(files[0]); LA = f0.latitude.values; LO = f0.longitude.values; f0.close()
    step = abs(float(LA[1]-LA[0]))
    jlat = np.round((LA[0]-lat)/step).astype(int)          # latitude DESCENDS from LA[0]
    jlon = np.round((lon-LO[0])/step).astype(int)
    assert np.allclose(LA[jlat], lat) and np.allclose(LO[jlon], lon), "ERA5 regrid mismatch"
    VARS = varlist or ["sd","swvl1","swvl2","swvl3","swvl4","tp","e","ro","smlt"]
    E = {v: np.full((nM,nC), np.nan, np.float32) for v in VARS}
    for fn in files:
        ds = xr.open_dataset(fn)
        mm = ds.valid_time.dt.year.values*12 + ds.valid_time.dt.month.values
        for v in VARS:
            if v not in ds: continue
            arr = ds[v].values
            for i, M in enumerate(mm):
                k = int(M - m_lo)
                if 0 <= k < nM: E[v][k] = arr[i][jlat, jlon]
        ds.close()
    return E

ERA5X_DIR = Path("era5_extra")
ERA5X_VARS = ["2m_temperature", "potential_evaporation",
              "soil_temperature_level_1", "soil_temperature_level_2",
              "soil_temperature_level_3", "soil_temperature_level_4",
              "surface_runoff", "sub_surface_runoff",
              "snow_density", "snow_cover", "snow_evaporation",
              "skin_reservoir_content",
              "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]

def era5x_download(out_dir, y0=2002, y1=2019):
    """Second retrieval: the drivers and thermal state the first pass omitted.
       grid MUST stay 0.5 -- that grid carries the X.5 coordinates the challenge cells
       sit on exactly, so the regrid is a lookup.  A 1.0 grid returns integer coords."""
    import cdsapi; c = cdsapi.Client(); os.makedirs(out_dir, exist_ok=True)
    for y in range(y0, y1+1):
        dst = os.path.join(out_dir, f"era5land_extra_{y}.nc")
        if os.path.exists(dst): continue
        c.retrieve(CDS_DATASET, {"product_type":["monthly_averaged_reanalysis"],
            "variable":ERA5X_VARS, "year":[str(y)], "month":[f"{m:02d}" for m in range(1,13)],
            "time":["00:00"], "area":[84,-180,-56,180], "grid":[0.5,0.5],
            "data_format":"netcdf", "download_format":"unarchived"}, dst)

E5D = {}
if USE_ERA5:
    _t = time.time(); E5 = era5_load(ERA5_DIR)
    if E5 is None:
        print("no ERA5 files found in ./era5 -- continuing without them"); USE_ERA5 = False
    else:
        _S = E5["sd"].copy()
        for k, th in zip(["swvl1","swvl2","swvl3","swvl4"], LAYER_M): _S = _S + th*E5[k]
        E5D = {"S_era": _S,                                       # modelled column storage
               "deep" : 0.72*E5["swvl3"] + 1.89*E5["swvl4"],      # below the surface layer
               "swe"  : E5["sd"],                                 # snow -- invisible to SPEI
               "bal"  : E5["tp"] + E5["e"] - E5["ro"],            # dS/dt ~ P - E - R
               "melt" : E5["smlt"], "ro": E5["ro"]}
        cov = float(np.isfinite(E5D["S_era"]).mean())
        print(f"ERA5 loaded in {time.time()-_t:.0f}s | {len(E5D)} derived panels | coverage {cov:.3f}")
        print(f"  {int(round((1-cov)*nC))} of {nC} cells lie outside the ERA5-Land mask (ocean/coast)")
        print("  latest month any ERA5 feature reads: t = target-1  -> causal by construction")

        # ---- second set: drivers and thermal state -------------------------------------
        _t = time.time()
        EX = era5_load(ERA5X_DIR, pattern="era5land_extra_*.nc",
                       varlist=["t2m","pev","stl1","stl2","stl3","stl4","sro","ssro",
                                "rsn","snowc","es","src","lai_hv","lai_lv"])
        if EX is None:
            print("  no era5_extra files -- continuing with the first set only")
        else:
            _eps = 1e-9
            _pev = EX["pev"]; _e = E5["e"]
            _frz = sum(th*(EX[f"stl{i}"] < 273.15).astype(np.float32)
                       for i, th in zip((1,2,3,4), LAYER_M)) / sum(LAYER_M)
            _rt  = EX["sro"] + EX["ssro"]
            E5D.update({
                "pev"     : _pev,                                  # atmospheric demand
                "edef"    : _pev - _e,                             # unmet demand: potential - actual
                "ef"      : np.where(np.abs(_pev) > _eps, _e/np.where(np.abs(_pev)>_eps,_pev,np.nan),
                                     np.nan).astype(np.float32),   # evaporative fraction
                "t2m"     : EX["t2m"],
                "stl_top" : EX["stl1"], "stl_deep": EX["stl4"],
                "frozen"  : _frz.astype(np.float32),               # frozen share of the soil column
                "rpart"   : np.where(_rt > _eps, EX["sro"]/np.where(_rt>_eps,_rt,np.nan),
                                     np.nan).astype(np.float32),   # fast vs slow runoff split
                "rsn"     : EX["rsn"], "snowc": EX["snowc"], "es": EX["es"],
                "src"     : EX["src"], "lai": EX["lai_hv"] + EX["lai_lv"]})
            RAW5 = {"frozen": _frz.astype(np.float32), "snowc": EX["snowc"]}
            print(f"  era5_extra loaded in {time.time()-_t:.0f}s | +{13} derived panels "
                  f"| coverage {float(np.isfinite(E5D['edef']).mean()):.3f}")
            print("  pev-e = moisture limitation | stl<273K = frozen ground | sro/(sro+ssro) = runoff split")
            del EX
else:
    print("ERA5 disabled")

# ===================== CELL 14 =====================
# ---- (v11) Global Drought Observatory: multi-source SPI and satellite fAPAR --------------
# Caches are built by gdo_prepare.py / gdo_prepare_spi.py from the raw GDO NetCDF files,
# which are served over plain HTTP (gdo_download.py). Both caches sit on the 1-degree
# lat 89.5..-89.5 / lon -179.5..179.5 grid -- the challenge grid -- so placement is a lookup.
GDOP = {}

def _gdo_lookup(z):
    """Row/col of every challenge cell on a descending-latitude 1-degree grid."""
    la0, lo0 = float(z["lat"][0]), float(z["lon"][0])
    i = np.round(la0 - lat).astype(int); j = np.round(lon - lo0).astype(int)
    assert i.min() >= 0 and i.max() < len(z["lat"]) and j.min() >= 0 and j.max() < len(z["lon"]), \
        "GDO grid mismatch"
    return i, j

def _gdo_panel(z, key, i, j):
    """[nM, nC] panel on the challenge cells; months outside the panel are dropped.
       NpzFile decompresses the whole array on every z[key] access, so read it once."""
    A = z[key]; a = np.full((nM, nC), np.nan, np.float32)
    for q, M in enumerate(z["months"].astype(int)):
        m = int(M) - m_lo
        if 0 <= m < nM: a[m] = A[q][i, j]
    return a

if USE_GDO:
    _t = time.time()
    try:
        _z1 = np.load(GDO_DIR/"gdo_1deg_monthly.npz")
        _i, _j = _gdo_lookup(_z1)
        _src = {k: _gdo_panel(_z1, k, _i, _j) for k in ("spg03", "spg06", "fapan") if k in _z1.files}
        _f2 = GDO_DIR/"spi_multi_1deg_monthly.npz"
        if _f2.exists():
            _z2 = np.load(_f2); _i2, _j2 = _gdo_lookup(_z2)
            _src.update({k: _gdo_panel(_z2, k, _i2, _j2) for k in ("spp03", "spp06", "spj06")
                         if k in _z2.files})
        def _multi(keys):
            """Cross-source mean and spread. Spread is NaN where fewer than two sources exist."""
            S = np.stack([_src[k] for k in keys if k in _src])
            n = np.isfinite(S).sum(axis=0)
            with np.errstate(all="ignore"):
                mu = np.where(n > 0, np.nanmean(S, axis=0), np.nan).astype(np.float32)
                sd = np.where(n > 1, np.nanstd(S, axis=0), np.nan).astype(np.float32)
            return mu, sd, [k for k in keys if k in _src]
        _m3, _s3, _k3 = _multi(("spg03", "spp03"))
        _m6, _s6, _k6 = _multi(("spg06", "spp06", "spj06"))
        GDOP = {"g_spg03": _src["spg03"],          # GPCC alone: v10's #3 feature at 10% of SHAP
                "g_spi3":  _m3, "g_spi3_spr": _s3,  # multi-source SPI-3 and disagreement
                "g_spi6":  _m6, "g_spi6_spr": _s6,  # multi-source SPI-6 and disagreement
                "g_fapan": _src["fapan"]}           # MODIS fAPAR anomaly
        GDOP = {k: v for k, v in GDOP.items() if np.isfinite(v).any()}
        print(f"GDO: {len(_src)} source panels -> {len(GDOP)} feature panels in {time.time()-_t:.0f}s")
        print(f"  SPI-3 sources {_k3} | SPI-6 sources {_k6}")
        print("  coverage on challenge cells: " +
              " ".join(f"{k} {np.isfinite(v).mean():.2f}" for k, v in GDOP.items()))
        if len(_k3) > 1:
            _a, _b = _src[_k3[0]], _src[_k3[1]]; _ok = np.isfinite(_a) & np.isfinite(_b)
            print(f"  cross-source agreement SPI-3 ({_k3[0]} vs {_k3[1]}): "
                  f"r = {np.corrcoef(_a[_ok], _b[_ok])[0,1]:.3f}")
        print("  every GDO value is an observation dated at month t or earlier; nothing forecast")
        del _src, _z1
    except Exception as _e:
        print("GDO skipped:", type(_e).__name__, _e); GDOP = {}
else:
    print("GDO disabled")

# ===================== CELL 15 =====================
def kalman(DEV, VCt, phif=0.5226, w_s=0.801, w_f=0.032, kL=0.004):
    """(E) Causal throughout: at month m both the variance scale VCt[m] and the per-cell
       memory/noise parameters come from months < m. The old version passed a single
       whole-record Vc and whole-record (phi, noise), which let the scoring era set the
       filter's gain from month zero."""
    nMl, n = DEV.shape
    PS, NO = cell_params_causal(DEV)
    pf  = np.full(n, phif)
    X   = np.zeros((3,n)); Pm = np.zeros((3,3,n))
    V0  = VCt[0]
    Pm[0,0] = V0*0.5; Pm[1,1] = V0*0.5; Pm[2,2] = V0*w_f
    XF = np.zeros((nMl,3,n), np.float32); PF = np.zeros_like(XF)
    for m in range(nMl):
        ps = PS[m]; noi = NO[m]; Vc = VCt[m]
        phi = np.stack([np.ones(n), ps, pf])
        sig = 1.0 - noi
        q   = np.stack([kL*Vc,
                        (w_s/(w_s+w_f))*sig*Vc*(1-ps**2),
                        (w_f/(w_s+w_f))*sig*Vc*(1-pf**2)])
        R   = noi*Vc
        X  = phi*X                                             # predict
        Pm = Pm*(phi[:,None,:]*phi[None,:,:])
        Pm[0,0]+=q[0]; Pm[1,1]+=q[1]; Pm[2,2]+=q[2]
        y  = DEV[m]; ok = np.isfinite(y)
        if ok.any():                                           # update where observed
            PH = Pm.sum(axis=1); S = PH.sum(axis=0)+R; K = PH/S
            X  = X + K*np.where(ok, y - X.sum(axis=0), 0.0)*ok
            Pm = Pm - np.where(ok[None,None,:], K[:,None,:]*PH[None,:,:], 0.0)
        XF[m] = X; PF[m] = np.stack([Pm[0,0],Pm[1,1],Pm[2,2]])
    return XF, PF, PS.astype(np.float32), NO.astype(np.float32), pf

class Ctx:
    """All per-cell statistics expanding and causal: value at month m uses months < m only."""
    def __init__(self):
        self.CLIM = np.zeros((nM,nC), np.float32); self.CLIM_N = np.zeros((nM,nC), np.float32)
        acc = np.zeros((12,nC)); cnt = np.zeros((12,nC))
        for m in range(nM):
            k = mon[m]
            self.CLIM[m]   = np.where(cnt[k] > 0, acc[k]/np.maximum(cnt[k],1), 0.0)
            self.CLIM_N[m] = cnt[k]
            v = TWS[m]; ok = np.isfinite(v); acc[k][ok] += v[ok]; cnt[k][ok] += 1
        self.DEV = (TWS - self.CLIM).astype(np.float64)
        s1 = np.zeros(nC); s2 = np.zeros(nC); n = np.zeros(nC)
        self.VC = np.zeros((nM,nC), np.float32)
        for m in range(nM):
            self.VC[m] = np.where(n >= 8, np.maximum(s2/np.maximum(n,1)-(s1/np.maximum(n,1))**2, 1e-4), np.nan)
            v = self.DEV[m]; ok = np.isfinite(v); s1[ok]+=v[ok]; s2[ok]+=v[ok]**2; n[ok]+=1
        # (E) fill the early-month gap from the FIT era only -- VC[-1] is the whole record
        _fill   = float(np.nanmedian(self.VC[:FIT_END_REG]))
        self.VC = np.nan_to_num(self.VC, nan=_fill)
        self.VCt = np.maximum(self.VC.astype(np.float64), 1e-4)
        self.Vc = self.VCt[-1]                      # reporting only, never fed to the filter
        self.XF, self.PF, self.PS, self.NO, self.pf = kalman(self.DEV, self.VCt)
        self.Rt = (self.NO * self.VCt).astype(np.float32)
        self.KSLOW_S = nan_gauss_panel(self.XF[:,1,:], 1.5)
        self.KLVL_S  = nan_gauss_panel(self.XF[:,0,:], 1.5)
        self.ANOM_gN, self.ANOM_gE, self.ANOM_lap = grid_ops(self.DEV.astype(np.float32))
        self.ANOM_S1 = nan_gauss_panel(self.DEV.astype(np.float32), 1.0)
        self.ANOM_S2 = nan_gauss_panel(self.DEV.astype(np.float32), 2.0)
        # spatial basis: denoised reconstructions of the anomaly field at 3 truncations
        self.EOF = eof_panels(self.DEV.astype(np.float32)) if USE_EOF else {}
        # (v9-A) eof_panels leaves a month NaN when fewer than 200 cells are observed. At
        # test time anchors are 5-6 months apart, so anc-1 is almost never reconstructed and
        # v7's four eof*_d features were populated in training but NaN in prediction.
        # Difference against the last month the basis actually reconstructed.
        self.EOF_PREV = np.arange(nM, dtype=np.int64)
        if self.EOF:
            _v = np.isfinite(self.EOF[max(EOF_K)]).any(axis=1); _last = 0
            for _m in range(nM):
                self.EOF_PREV[_m] = _last
                if _v[_m]: _last = _m
        self.FS = {k: nan_gauss_panel(P[k], 1.5) for k in FC}
        # (E) expanding per-cell std / lag-1 AC / count: the value at month m uses months < m.
        # These were whole-record aggregates, which contradicted the causality claim.
        d = self.DEV
        self.cell_std = np.zeros((nM,nC), np.float32)
        self.cell_ac1 = np.zeros((nM,nC), np.float32)
        self.cell_n   = np.zeros((nM,nC), np.float32)
        _s1 = np.zeros(nC); _s2 = np.zeros(nC); _n = np.zeros(nC)
        _xy = np.zeros(nC); _xx = np.zeros(nC); _prev = np.full(nC, np.nan)
        for m in range(nM):
            _mu = _s1/np.maximum(_n,1)
            self.cell_std[m] = np.sqrt(np.maximum(_s2/np.maximum(_n,1) - _mu**2, 0.0))
            self.cell_ac1[m] = np.clip(_xy/np.maximum(_xx,1e-6), 0, 0.99)
            self.cell_n[m]   = _n
            v = d[m]; ok = np.isfinite(v)
            ok2 = ok & np.isfinite(_prev)
            _xy[ok2] += _prev[ok2]*v[ok2]; _xx[ok2] += _prev[ok2]**2
            _s1[ok] += v[ok]; _s2[ok] += v[ok]**2; _n[ok] += 1
            _prev = v
        CS = np.nancumsum(np.nan_to_num(self.DEV), axis=0); CN = np.cumsum(np.isfinite(self.DEV), axis=0)
        self.LL = np.zeros((nM,nC), np.float32); self.LN = np.zeros((nM,nC), np.float32)
        for m in range(nM):
            lo = max(m-24, 0)
            s_ = CS[m] - (CS[lo] if lo > 0 else 0.0); n_ = CN[m] - (CN[lo] if lo > 0 else 0)
            self.LL[m] = np.where(n_ >= 6, s_/np.maximum(n_,1), 0.0); self.LN[m] = n_
        self.E5A = {k: exp_anom(v) for k, v in E5D.items()}
        _sm = [k for k in ("S_era", "deep", "edef", "frozen") if k in self.E5A]
        self.E5S = {k: nan_gauss_panel(self.E5A[k], 1.5) for k in _sm} if E5D else {}
        # Exact TreeSHAP on the previous model: the smoothed ERA5 storage field carries 71x
        # the attribution of the same quantity at the cell itself (17.5% vs 0.2%). The model
        # reads spatial structure, not point values -- so sample that structure at several
        # widths instead of one arbitrary sigma=1.5.
        self.E5M = {}
        if "S_era" in self.E5A:
            for _s in (0.5, 1.0, 2.5, 4.0): self.E5M[_s] = nan_gauss_panel(self.E5A["S_era"], _s)
        if "deep" in self.E5A: self.E5M["deep40"] = nan_gauss_panel(self.E5A["deep"], 4.0)
        if "S_era" in self.E5A:
            _gn, _ge, _lp = grid_ops(self.E5A["S_era"])
            self.E5_gN, self.E5_gE, self.E5_lap = _gn, _ge, _lp
        if "edef" in self.E5A: self.E5M["edef40"] = nan_gauss_panel(self.E5A["edef"], 4.0)
        # (v11) GDO panels. SPI is standardised per calendar month by construction and the
        # fAPAR product is already an anomaly, so no deseasonalisation is applied. The model
        # reads regional structure far harder than point values (e_S_s40 vs e_S_era: 71x),
        # so the sigma=4.0 companion is built for every panel and sigma=1.5 for the means.
        self.GDO_A = dict(GDOP)
        self.GDO_S40 = {k: nan_gauss_panel(v, 4.0) for k, v in GDOP.items()}
        self.GDO_S15 = {k: nan_gauss_panel(GDOP[k], 1.5) for k in ("g_spi3", "g_spi6", "g_fapan")
                        if k in GDOP}
        # absolute state, NOT deseasonalised: "is the ground frozen right now" is the signal
        self.RAW = {k: v for k, v in (RAW5.items() if "RAW5" in globals() else [])}
        # static per-cell hydrological character, computed on the FIT era only so no
        # information from the scoring period enters. Physical descriptors, not coordinates:
        # they transfer to cells never seen in training, which lat/lon cannot do.
        self.REG = {}
        if E5D and "pev" in E5D:
            _f = slice(0, FIT_END_REG)
            def _m(a):
                with np.errstate(all="ignore"): return np.nanmean(a[_f], axis=0)
            _P, _PE = _m(E5D.get("bal")), _m(E5D["pev"])
            self.REG["r_aridity"]  = np.where(np.abs(_PE) > 1e-9, _P/np.where(np.abs(_PE)>1e-9,_PE,np.nan), np.nan).astype(np.float32)
            self.REG["r_snowfrac"] = np.where(np.abs(_m(E5D["S_era"])) > 1e-9,
                                              _m(E5D["swe"])/np.where(np.abs(_m(E5D["S_era"]))>1e-9,_m(E5D["S_era"]),np.nan), np.nan).astype(np.float32)
            self.REG["r_frozen"]   = _m(E5D["frozen"]).astype(np.float32)
            self.REG["r_lai"]      = _m(E5D["lai"]).astype(np.float32)
            self.REG["r_seas"]     = np.nanstd(np.stack([np.nanmean(np.where(mon[:FIT_END_REG,None]==k,
                                     E5D["S_era"][:FIT_END_REG], np.nan), axis=0) for k in range(12)]), axis=0).astype(np.float32)

t = time.time(); ctx = Ctx()
print(f"filter + context built in {time.time()-t:.0f}s" + (f" | {len(ctx.E5A)} ERA5 panels deseasonalised" if ctx.E5A else ""))
_ps_last = ctx.PS[-1]
print(f"per-cell slow phi (last month): p05 {np.percentile(_ps_last,5):.3f}  median {np.median(_ps_last):.3f}"
      f"  p95 {np.percentile(_ps_last,95):.3f}")

# ===================== CELL 17 =====================
def build(ctx, anc, cel, h, avail=None, ft=False):
    h = np.asarray(h); hi = h.astype(np.int64); tm = anc + hi; f = {}
    L, S, F    = ctx.XF[anc,0,cel], ctx.XF[anc,1,cel], ctx.XF[anc,2,cel]
    pL, pS, pF = ctx.PF[anc,0,cel], ctx.PF[anc,1,cel], ctx.PF[anc,2,cel]
    ps_a = ctx.PS[anc,cel]                               # (E) memory as known AT the anchor
    ds = (ps_a**hi).astype(np.float32); df = (ctx.pf[cel]**hi).astype(np.float32)
    f["k_level"], f["k_slow"], f["k_fast"] = L, S, F
    f["k_slow_h"], f["k_fast_h"] = S*ds, F*df
    f["k_pred"]  = L + S*ds + F*df                       # the filter's h-step forecast
    f["k_var"]   = pL + pS*ds**2 + pF*df**2 + ctx.Rt[anc,cel]  # how stale the state is
    f["k_var_L"], f["k_var_S"] = pL, pS
    f["k_state_sum"]  = L + S + F
    f["k_phi_s"]      = ps_a.astype(np.float32)
    f["k_noise"]      = ctx.NO[anc,cel].astype(np.float32)
    f["k_decay_s"]    = ds
    f["k_slow_s"]     = gather(ctx.KSLOW_S, anc, cel)
    f["k_level_s"]    = gather(ctx.KLVL_S,  anc, cel)
    f["k_resid_anchor"] = ctx.DEV[anc,cel] - (L+S+F)
    ll = ctx.LL[anc,cel]; f["local_level"] = ll; f["ll_n"] = ctx.LN[anc,cel]
    ac1h = (ctx.cell_ac1[anc,cel]**hi).astype(np.float32); f["ac1_pow_h"] = ac1h
    ct = ctx.CLIM[tm,cel]
    f["clim_target"] = ct; f["clim_n"] = ctx.CLIM_N[tm,cel]; f["clim_delta"] = ct - ctx.CLIM[anc,cel]
    f["base"] = ct + f["k_pred"]                         # the analytic forecast the booster corrects
    f["base_ar"] = ct + ll + (ctx.DEV[anc,cel] - ll)*ac1h
    f["tws"]  = TWS[anc,cel]
    f["anom"] = ctx.DEV[anc,cel]
    f["anom_s1"] = gather(ctx.ANOM_S1, anc, cel); f["anom_s2"] = gather(ctx.ANOM_S2, anc, cel)
    f["anom_hf"] = f["anom"] - f["anom_s2"]              # local signal after the smooth field
    f["horizon"] = hi.astype(np.float32)
    f["cell_std"] = ctx.cell_std[anc,cel]; f["cell_ac1"] = ctx.cell_ac1[anc,cel]
    f["cell_n"]   = ctx.cell_n[anc,cel]
    # ---- spatial-basis (EOF) features ----------------------------------------------
    # The reconstruction is the field with its high-order components -- where the
    # observation noise lives -- removed. The residual is what the basis discarded.
    if ctx.EOF:
        _prev = ctx.EOF_PREV[anc]                       # (v9-A) last RECONSTRUCTED month
        f["eof_dt"] = (anc - _prev).astype(np.float32)   # how stale that difference is
        for _K in EOF_K:
            _E = ctx.EOF[_K]
            _cur = gather(_E, anc, cel)
            f[f"eof{_K}"]   = _cur                                  # denoised anomaly at t
            f[f"eof{_K}_d"] = _cur - gather(_E, _prev, cel)         # denoised 1-month change
        _fine = f[f"eof{max(EOF_K)}"]
        f["eof_resid"]  = f["anom"] - _fine                         # what truncation threw away
        f["eof_coarse"] = f[f"eof{min(EOF_K)}"] - _fine             # large-scale minus full
        # the basis-denoised state propagated h steps by the filter's own slow memory
        f["eof_pred"]   = ct + _fine*ds
    f["anom_gN"]  = gather(ctx.ANOM_gN,  anc, cel)      # (D) directional structure
    f["anom_gE"]  = gather(ctx.ANOM_gE,  anc, cel)
    f["anom_lap"] = gather(ctx.ANOM_lap, anc, cel)      # cell minus its 4 neighbours
    ang = 2*np.pi*mon[tm]/12
    f["month_sin"] = np.sin(ang).astype(np.float32); f["month_cos"] = np.cos(ang).astype(np.float32)
    navail = np.zeros(len(anc), np.float32)
    for col in FC:
        s = SH[col]; A = P[col]
        for k in range(1,7):                              # target-1 .. target-6 : all <= t
            v = gather(A, tm-k, cel)
            if avail is not None: v = np.where(avail[:,k-1] > 0, v, np.nan)
            f[f"{s}_l{k}"] = v
            if col == "SPEI_03_t": navail += np.isfinite(v)
        sm = gather(ctx.FS[col], tm-1, cel)
        f[f"{s}_s"]  = sm if avail is None else np.where(avail[:,0] > 0, sm, np.nan)
        f[f"{s}_d1"] = f[f"{s}_l1"] - f[f"{s}_l2"]
        f[f"{s}_d3"] = f[f"{s}_l1"] - f[f"{s}_l4"]
        acc = np.zeros(len(anc)); cnt = np.zeros(len(anc))
        for k in range(1,8):                              # mean over the anchor -> t gap
            v = np.where(k <= hi, gather(A, tm-k, cel), np.nan)
            if avail is not None and k <= 6: v = np.where(avail[:,k-1] > 0, v, np.nan)
            ok = np.isfinite(v); acc[ok] += v[ok]; cnt[ok] += 1
        f[f"{s}_gap"] = np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan).astype(np.float32)
        if ft:                                        # dated at t+1 -- see the flag above
            v = gather(A, tm, cel)
            if avail is not None and avail.shape[1] > 6:
                v = np.where(avail[:,6] > 0, v, np.nan)   # match the REAL test availability
            f[f"{s}_ft"], f[f"{s}_dft"] = v, v - f[f"{s}_l1"]
    if ft:
        f["has_ft"] = np.isfinite(f["sp3_ft"]).astype(np.float32)
    if USE_ERA5 and getattr(ctx, "E5A", None):
        EA = ctx.E5A                                     # all reads are at t = target-1 or earlier
        for k in ("S_era","deep","swe","bal","melt","ro"):
            f["e_"+k] = gather(EA[k], tm-1, cel)
        f["e_S_l2"] = gather(EA["S_era"], tm-2, cel)
        f["e_S_l3"] = gather(EA["S_era"], tm-3, cel)
        aS = gather(EA["S_era"], anc, cel)
        f["e_dS"]    = f["e_S_era"] - aS                 # storage change across the blind gap
        f["e_ddeep"] = f["e_deep"]  - gather(EA["deep"], anc, cel)
        f["e_dswe"]  = f["e_swe"]   - gather(EA["swe"],  anc, cel)
        f["e_dS1"]   = f["e_S_era"] - f["e_S_l2"]
        f["e_mismatch"] = (aS - ctx.DEV[anc,cel]).astype(np.float32)   # no deep aquifer in ERA5
        f["e_S_s"]    = gather(ctx.E5S["S_era"], tm-1, cel)
        f["e_deep_s"] = gather(ctx.E5S["deep"],  tm-1, cel)
        if getattr(ctx, "E5M", None):                    # the storage field at several widths
            for _s in (0.5, 1.0, 2.5, 4.0):
                f[f"e_S_s{int(_s*10):02d}"] = gather(ctx.E5M[_s], tm-1, cel)
            f["e_S_hf"]   = f["e_S_era"]  - f["e_S_s40"]   # cell minus its region
            f["e_S_grad"] = f["e_S_s05"]  - f["e_S_s40"]   # fine minus broad: spatial contrast
            if "deep40" in ctx.E5M: f["e_deep_s40"] = gather(ctx.E5M["deep40"], tm-1, cel)
            if getattr(ctx, "E5_lap", None) is not None:
                f["e_S_gN"]  = gather(ctx.E5_gN,  tm-1, cel)
                f["e_S_gE"]  = gather(ctx.E5_gE,  tm-1, cel)
                f["e_S_lap"] = gather(ctx.E5_lap, tm-1, cel)
            if "edef40" in ctx.E5M: f["x_edef_s40"] = gather(ctx.E5M["edef40"], tm-1, cel)
        acc = np.zeros(len(anc)); cnt = np.zeros(len(anc))
        for k in range(1,8):                             # water balance over the anchor -> t gap
            v = np.where(k <= hi, gather(EA["bal"], tm-k, cel), np.nan)
            o_ = np.isfinite(v); acc[o_] += v[o_]; cnt[o_] += 1
        f["e_bal_gap"] = np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan).astype(np.float32)
        if "edef" in EA:                                 # ---- second ERA5 set ----
            for k in ("pev","edef","ef","t2m","stl_top","stl_deep","rpart",
                      "rsn","src","lai","es","frozen","snowc"):
                f["x_"+k] = gather(EA[k], tm-1, cel)
            for k in ("frozen","snowc"):                  # absolute state, not anomaly
                if k in ctx.RAW: f[f"x_{k}_raw"] = gather(ctx.RAW[k], tm-1, cel)
            for k in ("edef","frozen","snowc","lai","t2m"):
                f["x_d_"+k] = f["x_"+k] - gather(EA[k], anc, cel)
            for k in ("edef","t2m","frozen"):             # mean across the blind gap
                acc = np.zeros(len(anc)); cnt = np.zeros(len(anc))
                for j in range(1,8):
                    v = np.where(j <= hi, gather(EA[k], tm-j, cel), np.nan)
                    o_ = np.isfinite(v); acc[o_] += v[o_]; cnt[o_] += 1
                f["x_gap_"+k] = np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan).astype(np.float32)
            for k in ("edef","frozen"):
                if k in ctx.E5S: f[f"x_{k}_s"] = gather(ctx.E5S[k], tm-1, cel)
            for k, v in ctx.REG.items(): f[k] = v[cel]   # static hydrological regime
    # ---- (v11) GDO observations: every read is at t = target-1 or at the anchor ----------
    # Not masked by `avail`: unlike the provided SPEI columns, GDO exists for every month,
    # including the ones Test.csv omits, so the same values are present at prediction time.
    if USE_GDO and getattr(ctx, "GDO_A", None):
        for k, v in ctx.GDO_A.items():
            f[k]         = gather(v, tm-1, cel)                    # latest observation <= t
            f[k+"_s40"]  = gather(ctx.GDO_S40[k], tm-1, cel)       # its regional field
            if k in ctx.GDO_S15: f[k+"_s15"] = gather(ctx.GDO_S15[k], tm-1, cel)
            if not k.endswith("_spr"):
                f[k+"_d"] = f[k] - gather(v, anc, cel)             # change across the blind gap
                acc = np.zeros(len(anc)); cnt = np.zeros(len(anc))
                for j in range(1, 8):                              # mean over the anchor -> t gap
                    vv = np.where(j <= hi, gather(v, tm-j, cel), np.nan)
                    o_ = np.isfinite(vv); acc[o_] += vv[o_]; cnt[o_] += 1
                f[k+"_gap"] = np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan).astype(np.float32)
        if "g_spi3" in f and "sp3_l1" in f:
            # supply-minus-demand: SPI is P only, the provided SPEI is P - PET. Their gap
            # isolates the evaporative-demand contribution to the deficit.
            f["g_spi3_minus_spei3"] = f["g_spi3"] - f["sp3_l1"]
    f["n_forcing"] = navail
    return f

_d = build(ctx, np.array([100,100]), np.array([5,6]), np.array([3,3], np.float32),
           ft=USE_TARGET_MONTH_FORCING)
FEATURES = list(_d)
NE5 = len([k for k in FEATURES if k.startswith("e_")])
NX5 = len([k for k in FEATURES if k.startswith("x_")])
NRG = len([k for k in FEATURES if k.startswith("r_")])
NGD = len([k for k in FEATURES if k.startswith("g_")])
print(f"{len(FEATURES)} features | ERA5 set1 {NE5} | ERA5 set2 {NX5} | regime {NRG}"
      f" | GDO {NGD} (v11) | no lat/lon and no per-cell mean")
print(np.array(FEATURES).reshape(-1,3) if len(FEATURES)%3==0 else FEATURES)

# ===================== CELL 19 =====================
def samples(m_from, m_to, per_month, r, reps=1):
    """(B) reps independent horizon draws per (anchor, cell) instead of one, over every
       observed cell rather than a 70% subsample. The (horizon, availability) pair is
       still drawn from the real test blocks, so the mix is unchanged -- only the count."""
    A,C,H,S = [],[],[],[]
    for a in range(12, nM-1):
        oc = np.where(np.isfinite(TWS[a]))[0]
        if len(oc) == 0: continue
        for _rep in range(reps):
            n  = min(per_month, len(oc)); cc = r.choice(oc, n, replace=False)
            hh, ss = draw(n, r); tm = a + hh.astype(int)
            k = (tm < nM) & (m_from <= tm) & (tm < m_to)
            if not k.any(): continue
            cc2,hh2,ss2,tm2 = cc[k],hh[k],ss[k],tm[k]
            k2 = np.isfinite(TWS[tm2,cc2])
            if not k2.any(): continue
            A.append(np.full(k2.sum(), a)); C.append(cc2[k2]); H.append(hh2[k2]); S.append(ss2[k2])
    return np.concatenate(A), np.concatenate(C), np.concatenate(H), np.concatenate(S)

def mat(a, c, h, s=None):
    f = build(ctx, a, c, h, avail=s, ft=USE_TARGET_MONTH_FORCING)
    X = np.empty((len(a), len(FEATURES)), np.float32)
    for i,k in enumerate(FEATURES): X[:,i] = f[k]
    return X

FIT_END, CAL_END = midx(2013,7), midx(2014,7)
FIT  = samples(0,       FIT_END, PER_MONTH, rng, REPS)
CAL  = samples(FIT_END, CAL_END, PER_MONTH, rng, REPS)
HOLD = samples(CAL_END, nM,      PER_MONTH, rng, REPS)
Xf, Xc, Xh = mat(*FIT), mat(*CAL), mat(*HOLD)
yf = TWS[FIT[0] +FIT[2].astype(int),  FIT[1]].astype(np.float64)
yc = TWS[CAL[0] +CAL[2].astype(int),  CAL[1]].astype(np.float64)
yh = TWS[HOLD[0]+HOLD[2].astype(int), HOLD[1]].astype(np.float64)
hc, hh_ = CAL[2].astype(int), HOLD[2].astype(int)
print(f"FIT {len(yf):,}  CAL {len(yc):,}  HOLD {len(yh):,}   ({time.time()-T0:.0f}s)")
print("HOLD horizon mix:", {h: round(float((hh_==h).mean()),3) for h in range(1,8)})

# ===================== CELL 21 =====================
def recency_w(tm):
    """(v9-B) Exponential recency weight on the TARGET month. v8's calibration failed because
       coefficients fitted on 2010-2015 did not transfer to the 2015-2018 test era; the same
       era gap applies to the boosters themselves, only more gently. Weighting recent targets
       up biases the fit toward the regime being predicted. Weights are normalised to mean 1
       so the effective sample size -- and therefore the round counts -- stay comparable."""
    if RECENCY_TAU <= 0: return None
    w = np.exp((np.asarray(tm, np.float64) - (nM-1))/RECENCY_TAU)
    return w/w.mean()

rmse = lambda a,b: float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
iB, iT, iCl = FEATURES.index("base"), FEATURES.index("tws"), FEATURES.index("clim_target")
bf, bc, bh = (np.nan_to_num(X[:,iB], nan=0.0) for X in (Xf,Xc,Xh))
ph = np.nan_to_num(Xh[:,iT],  nan=0.0)
ch = np.nan_to_num(Xh[:,iCl], nan=0.0)
RESULTS["climatology"] = rmse(yh,ch); RESULTS["persistence"] = rmse(yh,ph); RESULTS["kalman_base"] = rmse(yh,bh)
for k in ("climatology","persistence","kalman_base"): print(f"  {k:<14} {RESULTS[k]:.4f}")

PAR = dict(objective="l2", learning_rate=0.03, num_leaves=255, min_data_in_leaf=100,
           feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
           max_bin=(127 if FAST else 255),   # uint8 histograms halve the bin memory
           lambda_l2=2.0, verbose=-1, num_threads=NTHREAD)   # (G) finer bins, slower rate
try:
    from codecarbon import OfflineEmissionsTracker
    trk = OfflineEmissionsTracker(country_iso_code=os.environ.get("TWS_ISO", "CAN"),
                                  output_dir=str(OUT), log_level="error",
                                  save_to_file=True)
    trk.start()
except Exception:
    trk = None

def fit_predict(X, y, base, Xs, mode, nit, Xval=None, yval=None, bval=None, w=None):
    """Train on FIT. When a validation set is supplied the round count is chosen by early
       stopping on it rather than hardcoded -- CAL is used, never HOLD."""
    tgt = y if mode=="direct" else y-base
    d   = lgb.Dataset(X, tgt, feature_name=FEATURES, weight=w)
    vs  = None
    if Xval is not None:
        vt = yval if mode=="direct" else yval-bval
        vs = [lgb.Dataset(Xval, vt, feature_name=FEATURES, reference=d)]
    ps = [np.zeros(len(x)) for x in Xs]; bl = []; bi = []
    for s in SEEDS:
        cb = [lgb.early_stopping(60, verbose=False)] if vs else None
        b  = lgb.train({**PAR, "seed":s, "bagging_seed":s, "feature_fraction_seed":s},
                       d, nit, valid_sets=vs, callbacks=cb)
        bl.append(b); bi.append(int(b.best_iteration or nit))
        for i,x in enumerate(Xs): ps[i] += b.predict(x)/len(SEEDS)
    return ps, bl, bi

W_FIT = recency_w(FIT[0] + FIT[2].astype(int))
if W_FIT is not None:
    print(f"  (v9-B) recency weights on FIT: tau={RECENCY_TAU:.0f} mo | "
          f"p05 {np.percentile(W_FIT,5):.3f}  median {np.median(W_FIT):.3f}  p95 {np.percentile(W_FIT,95):.3f}")
keep = {}; BEST_IT = {}
for mode in ("direct","residual"):
    (pc,pv), bl, bi = fit_predict(Xf, yf, bf, [Xc,Xh], mode, MAXIT, Xc, yc, bc, w=W_FIT)
    if mode=="residual": pc, pv = pc+bc, pv+bh
    RESULTS["model_"+mode] = rmse(yh,pv); keep[mode] = (pc,pv,bl); BEST_IT[mode] = bi
    print(f"  LGBM {mode:<9} {RESULTS['model_'+mode]:.4f}   stopped at {bi} of {MAXIT}"
          f"   ({time.time()-T0:.0f}s)")

# The final refit sees ~1.4x more rows than FIT, so the round budget scales with it.
_mb = int(np.mean([v for vs_ in BEST_IT.values() for v in vs_]))
N_ITER_F = int(round(_mb * 1.4))
RESULTS["best_iter_mean"] = _mb; RESULTS["n_iter_final"] = N_ITER_F
print(f"  early stopping chose ~{_mb} rounds -> final refit uses {N_ITER_F}")

# ---- two more gradient-boosting families on the same residual target -------------------
ENS = {"lgb_direct": (keep["direct"][0],   keep["direct"][1]),
       "lgb_resid" : (keep["residual"][0], keep["residual"][1])}
XGB_PAR = dict(n_estimators=700, learning_rate=0.04, max_depth=9, subsample=0.8,
               colsample_bytree=0.75, reg_lambda=2.0, tree_method="hist",
               n_jobs=NTHREAD, random_state=1, verbosity=0)
CAT_PAR = dict(iterations=900, learning_rate=0.06, depth=8, l2_leaf_reg=3.0,
               random_seed=1, verbose=0, thread_count=NTHREAD)
if SMOKE: XGB_PAR["n_estimators"] = 60; CAT_PAR["iterations"] = 60
# (v11) the row cap is shared by both families, so it is defined before either try-block:
# in v9 it lived inside the XGBoost block, and an XGBoost import failure would have made
# CatBoost -- the largest stack weight -- fail silently with a misleading NameError.
_cap = int(os.environ.get("TWS_CAP", "2000000"))
_sub = (np.random.default_rng(5).permutation(len(yf))[:_cap] if len(yf) > _cap
        else slice(None))
try:
    if not USE_EXTRA: raise RuntimeError("skipped in FAST mode")
    import xgboost as xgb
    _mx = xgb.XGBRegressor(**XGB_PAR)
    _mx.fit(Xf[_sub], (yf-bf)[_sub], sample_weight=(None if W_FIT is None else W_FIT[_sub]))
    ENS["xgb"] = (_mx.predict(Xc)+bc, _mx.predict(Xh)+bh)
    RESULTS["model_xgb"] = rmse(yh, ENS["xgb"][1])
    print(f"  XGBoost        {RESULTS['model_xgb']:.4f}   ({time.time()-T0:.0f}s)"); del _mx
except Exception as e:
    print("  XGBoost skipped:", e)
try:
    if not USE_EXTRA: raise RuntimeError("skipped in FAST mode")
    from catboost import CatBoostRegressor
    _mc = CatBoostRegressor(**CAT_PAR)
    _mc.fit(Xf[_sub], (yf-bf)[_sub], sample_weight=(None if W_FIT is None else W_FIT[_sub]))
    ENS["cat"] = (_mc.predict(Xc)+bc, _mc.predict(Xh)+bh)
    RESULTS["model_cat"] = rmse(yh, ENS["cat"][1])
    print(f"  CatBoost       {RESULTS['model_cat']:.4f}   ({time.time()-T0:.0f}s)"); del _mc
except Exception as e:
    print("  CatBoost skipped:", e)

# ---- stack: non-negative weights fitted on CAL ONLY, never on HOLD ---------------------
from scipy.optimize import nnls
ENS_KEYS = list(ENS)
_Ac = np.column_stack([ENS[k][0] for k in ENS_KEYS])
_Ah = np.column_stack([ENS[k][1] for k in ENS_KEYS])
_w, _ = nnls(_Ac, yc)
if _w.sum() <= 1e-9: _w = np.ones(len(ENS_KEYS))
W_ENS = _w/_w.sum()
RESULTS["ens_weights"] = {k: float(v) for k, v in zip(ENS_KEYS, W_ENS)}
pc, pv = _Ac @ W_ENS, _Ah @ W_ENS
RESULTS["model_stack"] = RESULTS["model_blend"] = rmse(yh, pv)
MODE = "stack"; RESULTS["mode"] = MODE; boosters = keep["residual"][2]

# ---- (A) spatial smoothing of the predicted field --------------------------------------
# GRACE resolves ~300 km, about three cells here, so the truth carries almost no power at
# sub-2-cell scales. Predicting cell by cell and never re-imposing that leaves high-frequency
# spatial noise in the output. Smooth the predicted ANOMALY (prediction minus climatology)
# within each target month; (sigma, weight) are chosen on CAL and only then applied to HOLD.
cc_ = np.nan_to_num(Xc[:,iCl], nan=0.0)
tm_c = CAL[0]  + CAL[2].astype(int);  cel_c = CAL[1]
tm_h = HOLD[0] + HOLD[2].astype(int); cel_h = HOLD[1]

def smooth_field(pred, clim_t, tmv, celv, sigma, w):
    """Blend each prediction with the Gaussian-smoothed field of its own target month."""
    if w <= 0: return pred.copy()
    a   = pred - clim_t
    out = a.copy()
    gr  = np.zeros((NLAT,NLON), np.float32); ms = np.zeros((NLAT,NLON), np.float32)
    for m in np.unique(tmv):
        k = np.where(tmv == m)[0]
        if len(k) < 50: continue
        ii, jj = ilat[celv[k]], ilon[celv[k]]
        gr[:] = 0.0; ms[:] = 0.0
        np.add.at(gr, (ii, jj), a[k])          # duplicates average rather than overwrite
        np.add.at(ms, (ii, jj), 1.0)
        num = gaussian_filter(gr, sigma, mode=("nearest","wrap"))
        den = gaussian_filter(ms, sigma, mode=("nearest","wrap"))
        sm  = np.where(den[ii,jj] > 1e-6, (num/den)[ii,jj], a[k])
        out[k] = (1-w)*a[k] + w*sm
    return out + clim_t

_best = (rmse(yc, pc), 0.0, 0.0)
for _sg in (0.5, 0.75, 1.0, 1.5, 2.0):
    for _w in (0.2, 0.4, 0.6, 0.8, 1.0):
        _r = rmse(yc, smooth_field(pc, cc_, tm_c, cel_c, _sg, _w))
        if _r < _best[0]: _best = (_r, _sg, _w)
SM_RMSE, SM_SIGMA, SM_W = _best
RESULTS["smooth_sigma"] = SM_SIGMA; RESULTS["smooth_w"] = SM_W
print(f"\n  (A) spatial smoothing chosen on CAL: sigma={SM_SIGMA} w={SM_W}"
      f"  (CAL {rmse(yc,pc):.4f} -> {SM_RMSE:.4f})")
if SM_W > 0:
    pc = smooth_field(pc, cc_, tm_c, cel_c, SM_SIGMA, SM_W)
    pv = smooth_field(pv, ch,  tm_h, cel_h, SM_SIGMA, SM_W)
RESULTS["model_smoothed"] = rmse(yh, pv)
print(f"  HOLD after smoothing  {RESULTS['model_smoothed']:.4f}"
      f"  (was {RESULTS['model_stack']:.4f})")
print("\n  stack weights (CAL-fitted):",
      {k: round(float(v), 3) for k, v in zip(ENS_KEYS, W_ENS)})
print(f"  stacked ensemble  {RESULTS['model_stack']:.4f}")
_bs = min(RESULTS['model_direct'], RESULTS['model_residual'],
          RESULTS.get('model_xgb', 9), RESULTS.get('model_cat', 9))
print(f"  best single model {_bs:.4f}")
print(f"\n  chosen framing: {MODE}")

# ===================== CELL 23 =====================
if PRICE_FT:
    other = not USE_TARGET_MONTH_FORCING
    def mat_ft(a, c, h, s, flag):
        f = build(ctx, a, c, h, avail=s, ft=flag); nm = list(f)
        X = np.empty((len(a), len(nm)), np.float32)
        for i,k in enumerate(nm): X[:,i] = f[k]
        return X, nm
    Xf2, NM2 = mat_ft(*FIT, other); Xh2, _ = mat_ft(*HOLD, other)
    b2f = np.nan_to_num(Xf2[:, NM2.index("base")], nan=0.0)
    b2h = np.nan_to_num(Xh2[:, NM2.index("base")], nan=0.0)
    d2 = lgb.Dataset(Xf2, yf - b2f, feature_name=NM2)
    m2 = lgb.train({**PAR, "seed":1, "bagging_seed":1, "feature_fraction_seed":1}, d2, _mb)
    p2 = m2.predict(Xh2) + b2h
    d1 = lgb.Dataset(Xf, yf - bf, feature_name=FEATURES)
    m1 = lgb.train({**PAR, "seed":1, "bagging_seed":1, "feature_fraction_seed":1}, d1, _mb)
    p1 = m1.predict(Xh) + bh
    hasft = HOLD[3][:,6] > 0
    pon = p1 if USE_TARGET_MONTH_FORCING else p2
    print(f"  on rows that HAVE target-month forcing ({hasft.mean():.1%}): {rmse(yh[hasft], pon[hasft]):.4f}")
    print(f"  on rows that do NOT                    ({1-hasft.mean():.1%}): {rmse(yh[~hasft], pon[~hasft]):.4f}")
    RESULTS["ft_on"]  = rmse(yh, p1 if USE_TARGET_MONTH_FORCING else p2)
    RESULTS["ft_off"] = rmse(yh, p2 if USE_TARGET_MONTH_FORCING else p1)
    RESULTS["ft_cost"] = RESULTS["ft_off"] - RESULTS["ft_on"]
    print(f"  single-seed, target-month forcing ON   {RESULTS['ft_on']:.4f}")
    print(f"  single-seed, target-month forcing OFF  {RESULTS['ft_off']:.4f}")
    print(f"  cost of full causality compliance      {RESULTS['ft_cost']:+.4f} RMSE")
    print("")
    print("  this notebook is running with USE_TARGET_MONTH_FORCING =", USE_TARGET_MONTH_FORCING)
    ec = np.corrcoef(p1 - yh, p2 - yh)[0,1]
    print(f"  error correlation between the two variants: {ec:.4f}")
    del Xf2, Xh2, m1, m2
else:
    RESULTS['ft_on'] = RESULTS['ft_off'] = RESULTS['ft_cost'] = float('nan')
    print('  causality pricing skipped (FAST mode) -- see run_summary of the full run')

# ===================== CELL 25 =====================
# ---- (F) h=1 specialist -----------------------------------------------------------------
# h=1 is 33% of the test set and physically distinct: the anchor IS target-1, so persistence
# is strong and the correction is a different function. One dedicated model, blended in with
# a weight chosen on CAL.
_f1 = FIT[2].astype(int) == 1
SPEC_B = 0.0
if _f1.sum() > 5000:
    _d1 = lgb.Dataset(Xf[_f1], (yf-bf)[_f1], feature_name=FEATURES)
    _s1 = lgb.train({**PAR, "seed":1, "bagging_seed":1, "feature_fraction_seed":1}, _d1, _mb)
    _c1 = hc == 1; _h1 = hh_ == 1
    if _c1.sum() > 200:
        _pc1 = _s1.predict(Xc[_c1]) + bc[_c1]
        _best_b = (rmse(yc[_c1], pc[_c1]), 0.0)
        for _b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            _r = rmse(yc[_c1], (1-_b)*pc[_c1] + _b*_pc1)
            if _r < _best_b[0]: _best_b = (_r, _b)
        SPEC_B = _best_b[1]
        if SPEC_B > 0:
            pc[_c1] = (1-SPEC_B)*pc[_c1] + SPEC_B*_pc1
            _pv1 = _s1.predict(Xh[_h1]) + bh[_h1]
            pv[_h1] = (1-SPEC_B)*pv[_h1] + SPEC_B*_pv1
        print(f"  (F) h=1 specialist blend weight {SPEC_B} (CAL h=1 {_best_b[0]:.4f})")
RESULTS["spec_blend"] = SPEC_B
RESULTS["model_spec"] = rmse(yh, pv)
print(f"  HOLD after h=1 specialist  {RESULTS['model_spec']:.4f}")

coef = {}
for h in range(1,8):
    m = hc == h
    if m.sum() < 200: coef[h] = (1.0, 0.0); continue
    A = np.column_stack([pc[m], np.ones(int(m.sum()))])
    coef[h] = tuple(np.linalg.lstsq(A, yc[m], rcond=None)[0])
cal = np.array([coef[int(x)][0] for x in hh_])*pv + np.array([coef[int(x)][1] for x in hh_])
RESULTS["model_calibrated"] = rmse(yh, cal)
# (v9-C) MEASURED: per-horizon calibration cost +0.0044 on v8's holdout and +0.0040 on the
# leaderboard. Coefficients fitted on one era do not transfer to another. Off, unconditionally.
RESULTS["use_calibration"]  = (False if FORCE_NO_CAL else
                               bool(RESULTS["model_calibrated"] < RESULTS["model_spec"]))
base_hold = cal if RESULTS["use_calibration"] else pv
print(f"  pre-cal {RESULTS['model_spec']:.4f} | calibrated {RESULTS['model_calibrated']:.4f}"
      f" -> use={RESULTS['use_calibration']}")

# Latitude bands carry a residual additive bias (see section 12). The offset is estimated on
# CAL and adopted only if it also helps on HOLD.
BANDS = [(-90,-23.5), (-23.5,23.5), (23.5,50), (50,90)]
def band_id(la):
    la = np.asarray(la); b = np.zeros(len(la), np.int64)
    for i,(lo,hi) in enumerate(BANDS): b[(la > lo) & (la <= hi)] = i
    return b
cal_c   = np.array([coef[int(x)][0] for x in hc])*pc + np.array([coef[int(x)][1] for x in hc])
pc_used = cal_c if RESULTS["use_calibration"] else pc
bid_c, bid_h = band_id(lat[CAL[1]]), band_id(lat[HOLD[1]])
BBIAS = np.array([float((yc[bid_c==i] - pc_used[bid_c==i]).mean()) if (bid_c==i).sum() > 200
                  else 0.0 for i in range(len(BANDS))])
deb = base_hold + BBIAS[bid_h]
RESULTS["model_debiased"] = rmse(yh, deb)
# (v9-D) v7 selected this by comparing on HOLD, which is not a trustworthy signal, and the
# same family of post-hoc additive corrections is what cost v8 its 0.0040. Off.
RESULTS["use_band_bias"]  = (False if FORCE_NO_CAL else
                             bool(RESULTS["model_debiased"] < rmse(yh, base_hold)))
if not RESULTS["use_band_bias"]: BBIAS = np.zeros(len(BANDS))
final_hold = base_hold + BBIAS[bid_h]
print(f"  band bias {np.round(BBIAS,4).tolist()} -> use={RESULTS['use_band_bias']}"
      f"  ({RESULTS['model_debiased']:.4f})")

RESULTS["model_final"] = rmse(yh, final_hold)
print(f"\n  FINAL HOLD RMSE  {RESULTS['model_final']:.4f}")
print(f"  gain over persistence  {RESULTS['persistence']-RESULTS['model_final']:+.4f}")

ladder = pd.DataFrame([("seasonal climatology", RESULTS["climatology"]),
                       ("persistence", RESULTS["persistence"]),
                       ("kalman analytic base", RESULTS["kalman_base"]),
                       ("LightGBM, direct", RESULTS["model_direct"]),
                       ("LightGBM, residual on filter", RESULTS["model_residual"]),
                       ("XGBoost, residual", RESULTS.get("model_xgb", np.nan)),
                       ("CatBoost, residual", RESULTS.get("model_cat", np.nan)),
                       ("stacked ensemble (CAL weights)", RESULTS["model_stack"]),
                       ("+ spatial smoothing", RESULTS["model_smoothed"]),
                       ("+ h=1 specialist", RESULTS["model_spec"]),
                       ("+ per-horizon calibration", RESULTS["model_calibrated"]),
                       ("+ per-band de-bias (final)", RESULTS["model_final"])],
                      columns=["approach","hold_rmse"])
ladder.to_csv(OUT/"baseline_ladder.csv", index=False)
print("\n", ladder.to_string(index=False))

# ===================== CELL 27 =====================
rows = [(h, int((hh_==h).sum()), rmse(yh[hh_==h],ph[hh_==h]),
         rmse(yh[hh_==h],bh[hh_==h]), rmse(yh[hh_==h],final_hold[hh_==h])) for h in range(1,8)]
byh = pd.DataFrame(rows, columns=["horizon","n","persistence","kalman_base","model"])
byh["gain"] = byh.persistence - byh.model
byh.to_csv(OUT/"validation_by_horizon.csv", index=False)
print(byh.round(4).to_string(index=False))

la = lat[HOLD[1]]; err = final_hold - yh; br = []
for lo,hi,nm in [(-90,-23.5,"S extratropics"),(-23.5,23.5,"Tropics"),(23.5,50,"N midlat"),(50,90,"N high")]:
    m = (la>lo)&(la<=hi)
    if m.sum(): br.append((nm,int(m.sum()),rmse(yh[m],ph[m]),rmse(yh[m],final_hold[m]),float(err[m].mean())))
byb = pd.DataFrame(br, columns=["band","n","persistence","model","bias"])
byb["gain"] = byb.persistence - byb.model
byb.to_csv(OUT/"validation_by_band.csv", index=False)
print("\n", byb.round(4).to_string(index=False))
print("\n  cell share by band:", {nm: f"{100*((lat>lo)&(lat<=hi)).mean():.1f}%"
      for lo,hi,nm in [(-90,-23.5,'S extratrop'),(-23.5,23.5,'Tropics'),(23.5,50,'N midlat'),(50,90,'N high')]})

# ===================== CELL 29 =====================
try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = rng.choice(len(Xh), min(20000, len(Xh)), replace=False)
    Xh = np.ascontiguousarray(Xh)
    ex  = shap.TreeExplainer(boosters[0])
    sv  = ex.shap_values(Xh[idx])
    imp = pd.DataFrame({"feature":FEATURES, "mean_abs_shap":np.abs(sv).mean(0)}) \
            .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT/"feature_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, Xh[idx], feature_names=FEATURES, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(OUT/"shap_summary.png", dpi=110); plt.close()
    print(imp.head(15).to_string(index=False))
    sp = imp[imp.feature.str.endswith("_s") | imp.feature.isin(["anom_hf","anom_s1","anom_s2"])]
    kf_ = imp[imp.feature.str.startswith("k_")]
    tot = imp.mean_abs_shap.sum()
    print(f"\n  spatially-derived features: {100*sp.mean_abs_shap.sum()/tot:.1f}% of total |SHAP|")
    print(f"  Kalman state features:      {100*kf_.mean_abs_shap.sum()/tot:.1f}%")
    RESULTS["shap_spatial_share"] = float(sp.mean_abs_shap.sum()/tot)
except Exception as e:
    print("SHAP skipped:", e)
    imp = pd.DataFrame({"feature":FEATURES,
                        "gain":boosters[0].feature_importance("gain")}).sort_values("gain",ascending=False)
    imp.to_csv(OUT/"feature_importance.csv", index=False); print(imp.head(15).to_string(index=False))

# ===================== CELL 31 =====================
bias = pd.DataFrame({"lat":la, "horizon":hh_, "err":err, "abs_lat":np.abs(la)})
rep = bias.groupby(pd.cut(bias.abs_lat, [0,15,30,45,60,90])).agg(
        n=("err","size"), bias=("err","mean"), rmse=("err", lambda s: float(np.sqrt((s**2).mean()))))
rep.to_csv(OUT/"bias_report.csv")
print(rep.round(4).to_string())
print("\n  bias by horizon:")
print(bias.groupby("horizon").err.agg(["size","mean"]).round(4).to_string())

# ===================== CELL 33 =====================
import gc
for _v in ("Xf", "Xc", "Xf2", "Xh2"):        # peak-memory trim before XA is built
    if _v in globals(): del globals()[_v]
gc.collect()
ALL = samples(0, nM, PER_MONTH_F, np.random.default_rng(11), REPS_F)
XA  = mat(*ALL); yA = TWS[ALL[0]+ALL[2].astype(int), ALL[1]].astype(np.float64)
bA  = np.nan_to_num(XA[:,iB], nan=0.0)
print(f"ALL {len(yA):,} samples  ({time.time()-T0:.0f}s)")

W_ALL = recency_w(ALL[0] + ALL[2].astype(int))
final_boost = {}
for mode in ("direct", "residual"):
    tgtA = yA if mode == "direct" else yA - bA
    dA   = lgb.Dataset(XA, tgtA, feature_name=FEATURES, weight=W_ALL)
    final_boost[mode] = [lgb.train({**PAR, "seed":s, "bagging_seed":s, "feature_fraction_seed":s},
                                   dA, N_ITER_F) for s in SEEDS]
    print(f"  refit lgb {mode:<9} {len(SEEDS)} seed(s)  ({time.time()-T0:.0f}s)")
SCALE = 1.4                                   # ALL has ~1.4x the rows FIT had
final_other = {}
if "xgb" in ENS_KEYS:
    import xgboost as xgb
    _p = {**XGB_PAR, "n_estimators": int(XGB_PAR["n_estimators"]*SCALE)}
    _subA = (np.random.default_rng(6).permutation(len(yA))[:_cap] if len(yA) > _cap
             else slice(None))
    final_other["xgb"] = xgb.XGBRegressor(**_p).fit(
        XA[_subA], (yA-bA)[_subA], sample_weight=(None if W_ALL is None else W_ALL[_subA]))
    print(f"  refit xgb                  ({time.time()-T0:.0f}s)")
if "cat" in ENS_KEYS:
    from catboost import CatBoostRegressor
    _p = {**CAT_PAR, "iterations": int(CAT_PAR["iterations"]*SCALE)}
    final_other["cat"] = CatBoostRegressor(**_p).fit(
        XA[_subA], (yA-bA)[_subA], sample_weight=(None if W_ALL is None else W_ALL[_subA]))
    print(f"  refit cat                  ({time.time()-T0:.0f}s)")

test["c"] = cidx(test.lat.to_numpy(), test.lon.to_numpy())
obs = np.isfinite(TWS)
anc = np.empty(len(test), np.int64)
for i,(mm,cc) in enumerate(zip(test.m.values-m_lo, test.c.values)):
    a = mm
    while a >= 0 and not obs[a,cc]: a -= 1
    anc[i] = a
hT = (test.m.values - m_lo + 1 - anc).astype(np.float32)
print("test horizons:", {k:int((hT==k).sum()) for k in range(1,8)})

XT    = mat(anc, test.c.values, hT, None)
bT    = np.nan_to_num(XT[:,iB], nan=0.0)
PT = {"lgb_direct": np.mean([b.predict(XT) for b in final_boost["direct"]],   axis=0),
      "lgb_resid" : np.mean([b.predict(XT) for b in final_boost["residual"]], axis=0) + bT}
for k, m in final_other.items(): PT[k] = m.predict(XT) + bT
pred = sum(W_ENS[i]*PT[k] for i, k in enumerate(ENS_KEYS))
print("  applied stack weights:", {k: round(float(W_ENS[i]),3) for i,k in enumerate(ENS_KEYS)})

# (F) the h=1 specialist, refitted on ALL, applied to the test rows at h=1
if SPEC_B > 0:
    _a1 = ALL[2].astype(int) == 1
    _ds = lgb.Dataset(XA[_a1], (yA-bA)[_a1], feature_name=FEATURES,
                      weight=(None if W_ALL is None else W_ALL[_a1]))
    _sf = lgb.train({**PAR, "seed":1, "bagging_seed":1, "feature_fraction_seed":1}, _ds, N_ITER_F)
    _t1 = hT.astype(int) == 1
    if _t1.any():
        pred[_t1] = (1-SPEC_B)*pred[_t1] + SPEC_B*(_sf.predict(XT[_t1]) + bT[_t1])
    print(f"  applied h=1 specialist to {int(_t1.sum()):,} rows (w={SPEC_B})")

# (A) the same spatial smoothing, on the real test months
if SM_W > 0:
    _cT  = np.nan_to_num(XT[:,iCl], nan=0.0)
    _tmT = (test.m.values - m_lo + 1).astype(np.int64)
    pred = smooth_field(pred, _cT, _tmT, test.c.values, SM_SIGMA, SM_W)
    print(f"  applied spatial smoothing sigma={SM_SIGMA} w={SM_W}")

if RESULTS["use_calibration"]:
    hcl  = np.clip(hT.astype(int), 1, 7)
    pred = np.array([coef[h][0] for h in hcl])*pred + np.array([coef[h][1] for h in hcl])
if RESULTS["use_band_bias"]:
    pred = pred + BBIAS[band_id(lat[test.c.values])]

sub = pd.DataFrame({"ID": test.ID.values, "Target": pred.astype(np.float32)})
ss_path = DATA_DIR/"SampleSubmission.csv"
if ss_path.exists():                       # align to the organisers' order when it is provided
    ss  = pd.read_csv(ss_path, usecols=["ID"])
    sub = ss.merge(sub, on="ID", how="left")
assert len(sub) == len(test), f"row count {len(sub)} != {len(test)}"
assert sub.Target.notna().all() and int(sub.ID.duplicated().sum()) == 0
SUB_PATH = OUT/"submission_v11.csv"
sub.to_csv(SUB_PATH, index=False)
RESULTS["submission"] = str(SUB_PATH)
print(f"\nwrote {SUB_PATH}  rows={len(sub):,}  mean={sub.Target.mean():.4f}"
      f"  range {sub.Target.min():.3f}..{sub.Target.max():.3f}")
sub.head()

# ===================== CELL 35 =====================
if USE_ERA5 and "imp" in globals():
    vcol  = [c for c in imp.columns if c != "feature"][0]
    e5    = imp[imp.feature.str.startswith("e_")]
    share = 100*float(e5[vcol].sum())/float(imp[vcol].sum())
    RESULTS["era5_importance_share"] = share/100
    print(f"ERA5 features: {len(e5)} of {len(imp)}   {share:.1f}% of total importance ({vcol})")
    print()
    print(e5.head(10).to_string(index=False))
    print()
    print("  snow (e_swe, e_dswe) and deep soil (e_deep, e_ddeep) are the two stores the")
    print("  provided SPEI / near-surface soil moisture cannot represent at all.")
elif USE_ERA5:
    print("ERA5 enabled but the importance table is unavailable (SHAP cell did not run)")
else:
    print("ERA5 disabled for this run")

# ===================== CELL 37 =====================
if trk:
    try: RESULTS["co2_kg"] = float(trk.stop())
    except Exception: RESULTS["co2_kg"] = None
RESULTS["train_seconds"] = round(time.time()-T0, 1)
RESULTS["n_features"] = len(FEATURES)
RESULTS["n_features_era5"] = len([k for k in FEATURES if k.startswith("e_")])
RESULTS["n_features_gdo"]  = len([k for k in FEATURES if k.startswith("g_")])
if "imp" in globals():
    _vcol = [c for c in imp.columns if c != "feature"][0]
    _g = imp[imp.feature.str.startswith("g_")]
    RESULTS["gdo_importance_share"] = float(_g[_vcol].sum())/float(imp[_vcol].sum())
    print(f"(v11) GDO features: {len(_g)} of {len(imp)}  "
          f"{100*RESULTS['gdo_importance_share']:.1f}% of total importance")
    print(_g.head(8).to_string(index=False)); print()
json.dump(RESULTS, open(OUT/"run_summary.json","w"), indent=2, default=float)
co2 = RESULTS.get("co2_kg") or 0.0
print(f"""
  FINAL HELD-OUT RMSE      {RESULTS['model_final']:.4f}
  persistence              {RESULTS['persistence']:.4f}   (gain {RESULTS['persistence']-RESULTS['model_final']:+.4f})
  kalman base, no learning {RESULTS['kalman_base']:.4f}
  LightGBM direct          {RESULTS['model_direct']:.4f}
  LightGBM residual        {RESULTS['model_residual']:.4f}
  XGBoost                  {RESULTS.get('model_xgb', float('nan')):.4f}
  CatBoost                 {RESULTS.get('model_cat', float('nan')):.4f}
  stacked ensemble         {RESULTS['model_stack']:.4f}   weights {RESULTS['ens_weights']}
  practical floor (B)      {min(RESULTS['floor_B']):.4f} - {max(RESULTS['floor_B']):.4f}
  framing                  {RESULTS['mode']}   calibration={RESULTS['use_calibration']}   band_bias={RESULTS['use_band_bias']}
  features                 {RESULTS['n_features']}  ({RESULTS['n_features_era5']} from ERA5-Land, {RESULTS['n_features_gdo']} from GDO)
  target-month forcing     {USE_TARGET_MONTH_FORCING}   (cost of turning it off: {RESULTS.get('ft_cost', float('nan')):+.4f})
  wall clock               {RESULTS['train_seconds']:.0f}s on CPU, no GPU
  CodeCarbon               {co2*1000:.1f} g CO2eq
  submission               {RESULTS.get('submission')}
""")


# ===================== completion marker =====================
print("\nV11 RUN COMPLETE")
