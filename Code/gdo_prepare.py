"""Collapse the raw GDO downloads to one compact 1-degree monthly cache.

Input   ./gdo/{spg03,spg06,fapan}/*.nc          (~2.8 GB, mostly fAPAR)
Output  ./gdo/gdo_1deg_monthly.npz              (~15 MB)

spg03 / spg06   already 1 degree monthly on lat 89.5..-89.5, lon -179.5..179.5 -- the exact
                grid the challenge cells sit on, so this is a straight copy.
fapan           0.083 degree, 36 dekads per year, latitude ASCENDING. Collapsed by a 12x12
                block nanmean to 1 degree and by dekad-to-month nanmean, then flipped so
                latitude descends like the SPI products.

Everything here is an observation dated at month t or earlier; no forecast, no GRACE.
"""
import os, sys, glob, time
import numpy as np, xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
GDO  = os.path.join(ROOT, "gdo")
DST  = os.path.join(GDO, "gdo_1deg_monthly.npz")
LAT  = np.arange(89.5, -90.0, -1.0)      # 180, descending
LON  = np.arange(-179.5, 180.0, 1.0)     # 360
BLK  = 12                                # 0.083 deg -> 1 deg


def month_key(t):
    t = np.asarray(t, dtype="datetime64[M]")
    y = t.astype("datetime64[Y]").astype(int) + 1970
    m = t.astype(int) % 12 + 1
    return y*12 + m


def load_1deg(pattern, var):
    """SPI products: already on the target grid."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        ds = xr.open_dataset(f)
        a = ds[var].values
        if a.ndim == 4: a = a[:, 0]                      # drop the singleton band axis
        keys = month_key(ds.time.values)
        lat = ds.lat.values
        if lat[0] < lat[-1]: a = a[:, ::-1, :]           # force descending latitude
        for i, k in enumerate(keys):
            out[int(k)] = a[i].astype(np.float32)
        ds.close()
    return out


def load_fapar(pattern, var):
    """fAPAR: 0.083 deg / 10-daily -> 1 deg / monthly, one timestep at a time."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        ds = xr.open_dataset(f)
        keys = month_key(ds.time.values)
        asc = float(ds.lat.values[0]) < float(ds.lat.values[-1])
        acc, cnt = {}, {}
        for i, k in enumerate(keys):
            k = int(k)
            v = ds[var].isel(time=i).values.astype(np.float32)
            if v.shape != (180*BLK, 360*BLK):
                raise SystemExit(f"unexpected fAPAR shape {v.shape} in {f}")
            with np.errstate(all="ignore"):
                g = np.nanmean(v.reshape(180, BLK, 360, BLK), axis=(1, 3)).astype(np.float32)
            if asc: g = g[::-1]
            ok = np.isfinite(g)
            if k not in acc:
                acc[k] = np.zeros((180, 360), np.float32); cnt[k] = np.zeros((180, 360), np.float32)
            acc[k][ok] += g[ok]; cnt[k][ok] += 1
        for k in acc:
            out[k] = np.where(cnt[k] > 0, acc[k]/np.maximum(cnt[k], 1), np.nan).astype(np.float32)
        ds.close()
        print(f"    {os.path.basename(f)} -> {len(keys)} dekads, {len(acc)} months", flush=True)
    return out


def stack(d, months):
    a = np.full((len(months), 180, 360), np.nan, np.float32)
    for i, k in enumerate(months):
        if k in d: a[i] = d[k]
    return a


def main():
    t0 = time.time(); got = {}
    for name, loader, pat in (
            ("spg03", load_1deg,  os.path.join(GDO, "spg03", "*.nc")),
            ("spg06", load_1deg,  os.path.join(GDO, "spg06", "*.nc")),
            ("fapan", load_fapar, os.path.join(GDO, "fapan", "*.nc"))):
        if not glob.glob(pat):
            print(f"  {name}: no files, skipping"); continue
        print(f"  {name} ...", flush=True)
        got[name] = loader(pat, name)
        print(f"  {name}: {len(got[name])} months  ({time.time()-t0:.0f}s)", flush=True)
    if not got:
        print("nothing to do"); return 1
    months = sorted(set().union(*[set(d) for d in got.values()]))
    np.savez_compressed(DST, months=np.array(months, np.int64), lat=LAT, lon=LON,
                        **{k: stack(v, months) for k, v in got.items()})
    ym = lambda k: f"{(k-1)//12}-{(k-1)%12+1:02d}"
    print(f"\nwrote {DST}  ({os.path.getsize(DST)/1e6:.0f} MB)")
    print(f"  {len(months)} months  {ym(months[0])} .. {ym(months[-1])}")
    for k, v in got.items():
        a = stack(v, months)
        print(f"  {k:6s} coverage {np.isfinite(a).mean():.3f}  "
              f"mean {np.nanmean(a):+.3f}  sd {np.nanstd(a):.3f}")
    print(f"  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
