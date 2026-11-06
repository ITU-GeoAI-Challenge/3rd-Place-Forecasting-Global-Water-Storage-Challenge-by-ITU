"""Tier-1 ERA5-Land retrieval: the variables the first download skipped.

Source      Copernicus Climate Data Store, `reanalysis-era5-land-monthly-means`
Target      D:/Water_Competition/era5_extra/era5land_extra_<YEAR>.nc, 2002-2019
Size        ~20 MB per year, ~380 MB total
Grid        same request parameters that produced the existing era5land_*.nc,
            so the two sets share a grid and the existing regrid code applies unchanged.

Prediction-time compliance
--------------------------
ERA5-Land monthly means for month t are published via ERA5T with ~5-day latency, so
month t is available once month t closes -- the same footing as the organisers' own
SPEI_t and SOIL_MOISTURE_t, which likewise cannot be computed until t completes.
Nothing dated t+1 is ever read. ERA5-Land does not assimilate GRACE, so no TWS
information enters, directly or indirectly.

Resumable: already-downloaded years are skipped, so re-running after an interruption
picks up where it stopped.
"""
import os, sys, time
import cdsapi

OUT_DIR = "D:/Water_Competition/era5_extra"
Y0, Y1 = 2002, 2019
DATASET = "reanalysis-era5-land-monthly-means"

VARS = [
    "2m_temperature",                      # PET driver / aridity
    "potential_evaporation",
    "soil_temperature_level_1",            # frozen ground -> the +0.079 high-latitude bias
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "surface_runoff",                      # runoff split (only total was downloaded before)
    "sub_surface_runoff",
    "snow_density",                        # snow characterisation beyond depth-equivalent
    "snow_cover",
    "snow_evaporation",
    "skin_reservoir_content",              # interception storage
    "leaf_area_index_high_vegetation",     # vegetation water use
    "leaf_area_index_low_vegetation",
]

os.makedirs(OUT_DIR, exist_ok=True)
c = cdsapi.Client()
t0 = time.time()
done, failed = [], []

for y in range(Y0, Y1 + 1):
    dst = os.path.join(OUT_DIR, f"era5land_extra_{y}.nc")
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"  {y}  already present ({os.path.getsize(dst)/1e6:.1f} MB), skipping", flush=True)
        done.append(y)
        continue
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARS,
        "year": [str(y)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": [84, -180, -56, 180],          # matches the challenge grid extent
        "grid": [0.5, 0.5],   # MUST match the existing era5land_*.nc: the 0.5 grid carries
                              # the X.5 coordinates the challenge cells sit on exactly, so the
                              # regrid is a lookup. A 1.0 grid returns integer coords instead
                              # and would force interpolation.
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    try:
        print(f"  {y}  requesting...", flush=True)
        c.retrieve(DATASET, req, dst)
        print(f"  {y}  OK  {os.path.getsize(dst)/1e6:.1f} MB   "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        done.append(y)
    except Exception as e:
        print(f"  {y}  FAILED: {type(e).__name__}: {e}", flush=True)
        failed.append(y)

print(f"\ndownloaded {len(done)}/{Y1-Y0+1} years in {time.time()-t0:.0f}s")
if failed:
    print(f"failed years: {failed}  -- re-run this script to retry just those")
    sys.exit(1)
print("all years present in", OUT_DIR)
