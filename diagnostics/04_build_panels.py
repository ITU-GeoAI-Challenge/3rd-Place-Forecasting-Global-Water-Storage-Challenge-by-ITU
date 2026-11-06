import os
os.makedirs("d:/Water_4_9_26/Water_Competition/diagnostics/cache/",exist_ok=True)
import pandas as pd, numpy as np, xarray as xr, time, pickle
R="d:/Water_4_9_26/Water_Competition/"
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
t0=time.time()
tr=pd.read_csv(R+"Train.csv",usecols=["time","lat","lon","TWS_t","target"])
te=pd.read_csv(R+"Test.csv",usecols=["time","lat","lon","TWS_t"])
for d in (tr,te):
    d['t']=pd.to_datetime(d.time); d['mk']=d.t.dt.year*12+d.t.dt.month
cells=pd.concat([tr[['lat','lon']],te[['lat','lon']]]).drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
cells['ci']=np.arange(len(cells))
key=lambda la,lo: (np.round(la*2).astype(np.int64)*100000+np.round(lo*2).astype(np.int64))
cmap={k:i for k,i in zip(key(cells.lat.values,cells.lon.values),cells.ci.values)}
m_lo=int(min(tr.mk.min(),te.mk.min())); m_hi=int(max(tr.mk.max()+1,te.mk.max()+1))
nM,nC=m_hi-m_lo+1,len(cells)
print("cells",nC,"months",nM,m_lo,m_hi)
TWS=np.full((nM,nC),np.nan,np.float32)
def put(df,col,off):
    ci=np.array([cmap[k] for k in key(df.lat.values,df.lon.values)])
    mi=df.mk.values.astype(int)+off-m_lo
    ok=(mi>=0)&(mi<nM)&np.isfinite(df[col].values)
    TWS[mi[ok],ci[ok]]=df[col].values[ok]
put(tr,'TWS_t',0); put(tr,'target',1); put(te,'TWS_t',0)
print("TWS panel coverage", np.isfinite(TWS).mean(), "months with data",int((np.isfinite(TWS).sum(1)>200).sum()))

# ---- ERA5 -> 1deg on the challenge cells ----
V1=["tp","e","ro","smlt","sd","swvl1","swvl2","swvl3","swvl4"]; V2=["t2m","pev","stl1","snowc","sro","ssro"]
P={v:np.full((nM,nC),np.nan,np.float32) for v in V1+V2}
for y in range(2002,2020):
    A=xr.open_dataset(R+f"era5/era5land_{y}.nc"); B=xr.open_dataset(R+f"era5_extra/era5land_extra_{y}.nc")
    la=A.latitude.values; lo=A.longitude.values
    tv=pd.to_datetime(A.valid_time.values); mk=(tv.year*12+tv.month).values
    i=np.clip(np.round((la[0]-cells.lat.values)/0.5).astype(int),0,len(la)-1)
    j=np.clip(np.round((cells.lon.values-lo[0])/0.5).astype(int),0,len(lo)-1)
    # 2x2 average of the four 0.5deg cells inside each 1deg cell
    for src,vs in ((A,V1),(B,V2)):
        for v in vs:
            X=src[v].values
            acc=np.zeros((len(mk),nC),np.float64); n=np.zeros((len(mk),nC))
            for di in (0,1):
                for dj in (0,1):
                    q=X[:,np.clip(i+di,0,len(la)-1),:][:,:,np.clip(j+dj,0,len(lo)-1)] if False else X[:,np.clip(i+di,0,len(la)-1),np.clip(j+dj,0,len(lo)-1)]
                    g=np.isfinite(q); acc[g]+=q[g]; n[g]+=1
            val=np.where(n>0,acc/np.maximum(n,1),np.nan)
            for q,k in enumerate(mk):
                m=int(k)-m_lo
                if 0<=m<nM: P[v][m]=val[q]
    A.close(); B.close()
    print(y,f"{time.time()-t0:.0f}s",end=" ",flush=True)
print()
np.savez_compressed(S+"panels.npz",TWS=TWS,m_lo=m_lo,nM=nM,nC=nC,
                    lat=cells.lat.values,lon=cells.lon.values,**{("e_"+k):v for k,v in P.items()})
print("saved",time.time()-t0,"s")
