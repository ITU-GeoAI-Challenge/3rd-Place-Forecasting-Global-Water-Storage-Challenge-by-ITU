import os
os.makedirs("d:/Water_4_9_26/Water_Competition/diagnostics/cache/",exist_ok=True)
import numpy as np, pandas as pd
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
R="d:/Water_4_9_26/Water_Competition/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); lat=z["lat"]; lon=z["lon"]; nC=len(lat)
key=lambda la,lo:(np.round(la*2).astype(np.int64)*100000+np.round(lo*2).astype(np.int64))
cmap={k:i for i,k in enumerate(key(lat,lon))}
COLS=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
P={c:np.full((nM,nC),np.nan,np.float32) for c in COLS}
for fn in (R+"Train.csv",R+"Test.csv"):
    d=pd.read_csv(fn,usecols=["time","lat","lon"]+COLS)
    t=pd.to_datetime(d.time); mk=(t.dt.year*12+t.dt.month).values-m_lo
    ci=np.array([cmap[k] for k in key(d.lat.values,d.lon.values)])
    ok=(mk>=0)&(mk<nM)
    for c in COLS:
        v=d[c].values; g=ok&np.isfinite(v)
        P[c][mk[g],ci[g]]=v[g]
    print(fn,"done")
np.savez_compressed(S+"spei.npz",**{("p_"+k):v for k,v in P.items()})
print({k:round(float(np.isfinite(v).mean()),3) for k,v in P.items()})
