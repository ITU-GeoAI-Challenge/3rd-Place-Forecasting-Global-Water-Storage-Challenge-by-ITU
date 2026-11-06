"""GDO (Global Drought Observatory) retrieval -- the covariates ERA5-Land cannot supply.

Source      Copernicus Global Drought Observatory, open HTTP, no authentication
            https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/
Target      ./gdo/<product>/<file>.nc , 2002-2019
Licence     CC-BY 4.0 (JRC) -- attribute in the report

Why these four
--------------
fapan   fAPAR anomaly, MODIS.  The vegetation channel is currently EMPTY: ERA5-Land's LAI is
        a fixed monthly climatology with no interannual variation, which is why `x_lai` earns
        0.0009 mean|SHAP and `x_d_lai` 0.0002. MODIS fAPAR anomaly is satellite-OBSERVED and
        leads storage drawdown.  MODIS, not VIIRS: VIIRS only starts ~2012 and would be absent
        from the whole training period.
smang   Soil moisture anomaly from LISFLOOD -- a different land-surface model to ERA5-Land's
        HTESSEL, so its errors are partly independent. Two independent estimates of the same
        state beat either alone.
spg03   SPI at 3 and 6 months, GPCC gauge-based. Precipitation ONLY, where the provided SPEI
spg06   is precipitation-minus-PET, so the pair separates supply from demand. Native 1 degree
        monthly -- lands on the challenge grid with no regridding.

Prediction-time compliance
--------------------------
All four are observational products dated at month t or earlier; nothing dated t+1 is read.
None of them ingest GRACE, directly or indirectly, so no TWS information leaks in.

Resumable: files already present are skipped.
"""
import os, sys, time, urllib.request, urllib.error

BASE = ("https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdo")
Y0, Y1 = 2002, 2019

# product -> (directory, version, filename template)
PRODUCTS_ALL = {
    "spg03": ("GDO_GPCC_Standardized_Precipitation_Index_SPI3", "ver1-2-0",
              "spg03_m_wld_{y}0101_{y}1201_m.nc"),
    "spg06": ("GDO_GPCC_Standardized_Precipitation_Index_SPI6", "ver1-2-0",
              "spg06_m_wld_{y}0101_{y}1201_m.nc"),
    "smang": ("GDO_Soil_Moisture_Index_Anomaly", "ver3-0-1",
              "smang_m_gdo_{y}0101_{y}1221_t.nc"),
    "fapan": ("GDO_Fraction_of_Absorbed_Photosynthetically_Active_Radiation_Anomalies_fAPAR_MODIS",
              "ver1-3-1", "fapan_m_wld_{y}0101_{y}1221_t.nc"),
    # Multi-source SPI. spg03 (GPCC) became the #3 feature overall at 10% of SHAP attribution,
    # so more INDEPENDENT precipitation observations is the direction the evidence supports.
    # CPC is gauge-based, JRA-3Q an independent reanalysis -- both distinct from GPCC.
    # ERA5-SPI is deliberately omitted: ERA5-Land precipitation is already in the model.
    # CHIRPS is omitted on size (6.5-19 GB at 0.05 deg).
    "spp06": ("GDO_CPC_Standardized_Precipitation_Index_SPI6", "ver1-0-0",
              "spp06_m_gdo_{y}0101_{y}1201_m.nc"),
    "spp03": ("GDO_CPC_Standardized_Precipitation_Index_SPI3", "ver1-0-0",
              "spp03_m_gdo_{y}0101_{y}1221_t.nc"),
    "spj06": ("GDO_JRA-3Q_Standardized_Precipitation_Index_SPI6", "ver1-0-0",
              "spj06_m_gdo_{y}0101_{y}1201_m.nc"),
}
# order matters: cheap+unique first, the 12 GB one last so it can be interrupted safely.
_ONLY = [x for x in os.environ.get("GDO_ONLY", "spg03,spg06,fapan,smang").split(",") if x]
PRODUCTS = {k: PRODUCTS_ALL[k] for k in _ONLY if k in PRODUCTS_ALL}


def fetch(url, dst, tries=3):
    """Stream in chunks -- smang years are ~685 MB and must not be buffered in memory."""
    for k in range(tries):
        try:
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    f.write(chunk)
            os.replace(tmp, dst)
            return os.path.getsize(dst)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return -1                      # that year genuinely is not published
            time.sleep(2 + 3*k)
        except Exception:
            time.sleep(2 + 3*k)
    return 0


def main():
    t0 = time.time(); total = 0; missing = []
    for name, (d, ver, tmpl) in PRODUCTS.items():
        sub = os.path.join(OUT_DIR, name)
        os.makedirs(sub, exist_ok=True)
        got = 0
        for y in range(Y0, Y1 + 1):
            fn = tmpl.format(y=y)
            dst = os.path.join(sub, fn)
            if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
                got += 1; total += os.path.getsize(dst); continue
            n = fetch(f"{BASE}/{d}/{ver}/{fn}", dst)
            if n > 0:
                got += 1; total += n
            else:
                if os.path.exists(dst):
                    os.remove(dst)
                missing.append(f"{name}/{y}")
        print(f"  {name:6s} {got:2d}/{Y1-Y0+1} years", flush=True)
    print(f"\n{total/1e6:.0f} MB in {time.time()-t0:.0f}s -> {OUT_DIR}")
    if missing:
        print("missing:", ", ".join(missing))
        print("re-run to retry just those")
        return 1
    print("all products complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
