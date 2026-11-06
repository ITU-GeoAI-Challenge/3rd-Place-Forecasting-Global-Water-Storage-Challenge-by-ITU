import os
os.makedirs("d:/Water_4_9_26/Water_Competition/diagnostics/cache/",exist_ok=True)
import numpy as np, pandas as pd
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
R="d:/Water_4_9_26/Water_Competition/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); lat=z["lat"]; lon=z["lon"]
TP=z["e_tp"]; PEV=z["e_pev"]
mon=np.array([(m_lo+m-1)%12 for m in range(nM)])
def anom(A):
    B=np.full_like(A,np.nan)
    for c in range(12):
        k=mon==c; B[k]=A[k]-np.nanmean(A[k],axis=0)
    return B
TPa=anom(TP); Da=anom(TP+PEV)
key=lambda la,lo:(np.round(la*2).astype(np.int64)*100000+np.round(lo*2).astype(np.int64))
cmap={k:i for i,k in enumerate(key(lat,lon))}
for tag,fn,cols in [("TEST",R+"Test.csv",["time","lat","lon","SPEI_01_t","SPEI_03_t","SOIL_MOISTURE_t"]),
                    ("TRAIN",R+"Train.csv",["time","lat","lon","SPEI_01_t","SPEI_03_t","SOIL_MOISTURE_t"])]:
    d=pd.read_csv(fn,usecols=cols)
    d['t']=pd.to_datetime(d.time); d['mk']=(d.t.dt.year*12+d.t.dt.month)-m_lo; d['mo']=d.t.dt.month
    ci=np.array([cmap[k] for k in key(d.lat.values,d.lon.values)])
    mk=d.mk.values.astype(int)
    print(f"\n=== {tag}  n={len(d)} ===")
    print(f"{'column':<18}"+"".join(f"{f't{s:+d}' if s else 't':>8}" for s in range(-4,2)))
    for col,A,k in [("SPEI_01_t",TPa,1),("SPEI_03_t",TPa,3),("SOIL_MOISTURE_t",TPa,1)]:
        a=d[col].values
        # anomaly of provided column per cell x calendar month
        df=pd.DataFrame({"v":a,"c":ci,"m":d.mo.values})
        aa=(df.v-df.groupby(["c","m"]).v.transform("mean")).values
        row=[]
        for sh in range(-4,2):
            acc=np.zeros(len(d)); n=np.zeros(len(d))
            for q in range(k):
                m2=mk+sh-q; ok=(m2>=0)&(m2<nM)
                v=np.full(len(d),np.nan); v[ok]=A[m2[ok],ci[ok]]
                g=np.isfinite(v); acc[g]+=v[g]; n[g]+=1
            b=np.where(n==k,acc,np.nan)
            df2=pd.DataFrame({"v":b,"c":ci,"m":d.mo.values})
            bb=(df2.v-df2.groupby(["c","m"]).v.transform("mean")).values
            ok=np.isfinite(aa)&np.isfinite(bb)
            row.append(np.corrcoef(aa[ok],bb[ok])[0,1])
        print(f"{col:<18}"+"".join(f"{x:>8.3f}" for x in row))
