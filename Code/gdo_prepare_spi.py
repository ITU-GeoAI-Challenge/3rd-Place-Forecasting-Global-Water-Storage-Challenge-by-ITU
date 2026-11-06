"""Regrid the multi-source SPI products to the challenge's 1-degree monthly grid.

Input   ./gdo/{spp03,spp06,spj06}/*.nc
Output  ./gdo/spi_multi_1deg_monthly.npz

Why these exist
---------------
`g_spg03_s40` -- SPI-3 from GPCC, smoothed at sigma=4.0 -- became the #3 feature overall in
v10 at 10% of total SHAP attribution, beating ECMWF SEAS5 by 6.5x. The lesson was that an
independent OBSERVATION of precipitation deficit at time t beats a FORECAST of t+1. So add
more independent observations of the same quantity:

    spg03/spg06   GPCC     gauge analysis        (already held, native 1 degree)
    spp03/spp06   CPC      independent gauge network, 0.5 deg
    spj06         JRA-3Q   independent reanalysis (NOT ERA5), 0.375 deg

The three disagree where precipitation is poorly observed -- sparse gauges, complex terrain,
high latitudes. That disagreement is itself information, so a cross-source spread is emitted
alongside the values.

Regridding
----------
0.5 deg divides into 1 deg cleanly; 0.375 deg does not (480 rows over 180 degrees). Rather
than special-case each, every source cell is assigned to the 1-degree cell its CENTRE falls
in and the members are averaged. Equivalent to a block mean where the factor is integral, and
well-defined where it is not.

Dekad products (`_t`, 36 steps/year) are collapsed to monthly by nanmean first.
"""
import os, sys, glob, time
import numpy as np, xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
GDO = os.path.join(ROOT, "gdo")
DST = os.path.join(GDO, "spi_multi_1deg_monthly.npz")
LAT = np.arange(89.5, -90.0, -1.0)       # 180, descending, matches the GPCC products
LON = np.arange(-179.5, 180.0, 1.0)      # 360

PRODUCTS = {"spp03": "spp03", "spp06": "spp06", "spj06": "spj06"}


def month_key(t):
    t = np.asarray(t, dtype="datetime64[M]")
    return (t.astype("datetime64[Y]").astype(int) + 1970)*12 + (t.astype(int) % 12 + 1)


def bin_index(src, edges_lo, step, n):
    """Target row/col for every source centre."""
    i = np.floor((edges_lo - src)/step).astype(int) if step > 0 else None
    return np.clip(i, 0, n-1)


def regrid(a, la, lo):
    """Source grid -> 1 degree by averaging the source centres falling in each target cell."""
    ii = np.clip(np.floor((90.0 - la)).astype(int), 0, 179)      # lat 90..-90 -> row 0..179
    jj = np.clip(np.floor((lo + 180.0)).astype(int), 0, 359)     # lon -180..180 -> col 0..359
    num = np.zeros((180, 360), np.float64); den = np.zeros((180, 360), np.float64)
    ok = np.isfinite(a)
    np.add.at(num, (ii[:, None].repeat(len(lo), 1)[ok], jj[None, :].repeat(len(la), 0)[ok]),
              a[ok])
    np.add.at(den, (ii[:, None].repeat(len(lo), 1)[ok], jj[None, :].repeat(len(la), 0)[ok]),
              1.0)
    return np.where(den > 0, num/np.maximum(den, 1), np.nan).astype(np.float32)


def load(name, var):
    out = {}
    for f in sorted(glob.glob(os.path.join(GDO, name, "*.nc"))):
        ds = xr.open_dataset(f)
        la = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
        lo = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values
        keys = month_key(ds.time.values)
        acc, cnt = {}, {}
        for i, k in enumerate(keys):
            k = int(k)
            v = ds[var].isel(time=i).values
            if v.ndim == 3: v = v[0]                      # drop the singleton band axis
            g = regrid(np.asarray(v, np.float32), la, lo)
            if k not in acc:
                acc[k] = np.zeros((180, 360), np.float64); cnt[k] = np.zeros((180, 360), np.float64)
            m = np.isfinite(g)
            acc[k][m] += g[m]; cnt[k][m] += 1
        for k in acc:
            out[k] = np.where(cnt[k] > 0, acc[k]/np.maximum(cnt[k], 1), np.nan).astype(np.float32)
        ds.close()
        print(f"    {os.path.basename(f)} -> {len(keys)} steps, {len(acc)} months", flush=True)
    return out


def main():
    t0 = time.time(); got = {}
    for name, var in PRODUCTS.items():
        if not glob.glob(os.path.join(GDO, name, "*.nc")):
            print(f"  {name}: no files, skipping"); continue
        print(f"  {name} ...", flush=True)
        got[name] = load(name, var)
        print(f"  {name}: {len(got[name])} months ({time.time()-t0:.0f}s)", flush=True)
    if not got:
        print("nothing to do"); return 1
    months = sorted(set().union(*[set(d) for d in got.values()]))
    def stack(d):
        a = np.full((len(months), 180, 360), np.nan, np.float32)
        for i, k in enumerate(months):
            if k in d: a[i] = d[k]
        return a
    arrs = {k: stack(v) for k, v in got.items()}
    np.savez_compressed(DST, months=np.array(months, np.int64), lat=LAT, lon=LON, **arrs)
    ym = lambda k: f"{(k-1)//12}-{(k-1)%12+1:02d}"
    print(f"\nwrote {DST} ({os.path.getsize(DST)/1e6:.0f} MB)")
    print(f"  {len(months)} months {ym(months[0])} .. {ym(months[-1])}")
    for k, a in arrs.items():
        print(f"  {k:6s} coverage {np.isfinite(a).mean():.3f}  mean {np.nanmean(a):+.3f}"
              f"  sd {np.nanstd(a):.3f}")
    print(f"  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
