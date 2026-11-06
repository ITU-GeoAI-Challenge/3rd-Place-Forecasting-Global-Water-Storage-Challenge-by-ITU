import os
os.makedirs("d:/Water_4_9_26/Water_Competition/diagnostics/cache/",exist_ok=True)
import numpy as np, netCDF4 as nc, glob, time
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
R="d:/Water_4_9_26/Water_Competition/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); lat=z["lat"]; lon=z["lon"]; nC=len(lat)
SP=np.full((nM,nC),np.nan,np.float32); ST=np.full((nM,nC),np.nan,np.float32); SD=np.full((nM,nC),np.nan,np.float32)
t0=time.time()
for f in sorted(glob.glob(R+"seas5/seas5_*.nc")):
    d=nc.Dataset(f)
    print(f, {k:v for k,v in d.variables['tprate'].__dict__.items() if k=='_ChunkSizes'},
          d.variables['tprate'].shape, flush=True)
    la=d.variables['latitude'][:].data; lo=d.variables['longitude'][:].data
    i=np.round((la[0]-lat)).astype(int); j=np.round((lon-lo[0])).astype(int)
    tvar=d.variables['forecast_reference_time']
    tt=nc.num2date(tvar[:],tvar.units,only_use_cftime_datetimes=False)
    nmem=d.dimensions['number'].size
    for v,dstm,dsts in (("tprate",SP,SD),("t2m",ST,None)):
        X=d.variables[v]
        cn=X.chunking()[0] if isinstance(X.chunking(),list) else nmem   # member chunk width
        s=None; s2=None; cnt=None
        for n0 in range(0,nmem,cn):
            n1=min(n0+cn,nmem)
            a=np.asarray(X[n0:n1,:,0,:,:],np.float64)                   # chunk-aligned read
            # SEAS5 is a 25-member hindcast to 2016 and 51-member from 2017: the extra members
            # are fill/NaN on the early initialisations, so accumulate NaN-aware.
            g=np.isfinite(a); a0=np.where(g,a,0.0)
            b=a0.sum(0); k=g.sum(0)
            s = b if s is None else s+b
            cnt = k if cnt is None else cnt+k
            if dsts is not None:
                c=(a0*a0).sum(0); s2 = c if s2 is None else s2+c
            del a,a0,g
            print(f"   {v} members {n0}:{n1} {time.time()-t0:.0f}s",flush=True)
        cnt=np.maximum(cnt,1)
        mu=np.where(cnt>0,s/cnt,np.nan)
        sd=np.sqrt(np.maximum(s2/cnt-mu*mu,0)) if dsts is not None else None
        print(f"   {v} members used per init: min {cnt.min()} max {cnt.max()}",flush=True)
        for k in range(len(tt)):
            M=(tt[k].year*12+tt[k].month); m=M+1-m_lo
            if 0<=m<nM:
                dstm[m]=mu[k][i,j]
                if dsts is not None: dsts[m]=sd[k][i,j]
    d.close()
np.savez_compressed(S+"seas5.npz",s_tp=SP,s_t2m=ST,s_tp_sd=SD)
print("coverage",np.isfinite(SP).mean(),f"{time.time()-t0:.0f}s")
mon=np.array([(m_lo+m-1)%12 for m in range(nM)])
def anom(A):
    B=np.full_like(A,np.nan)
    for c in range(12):
        k=mon==c
        with np.errstate(all="ignore"): B[k]=A[k]-np.nanmean(A[k],axis=0)
    return B
TPa=anom(z["e_tp"]); T2a=anom(z["e_t2m"]); SPa=anom(SP); STa=anom(ST)
print("\nSEAS5 forecastMonth=2 (lead 1 month) anomaly-correlation skill vs ERA5-Land")
print(f"{'band':<22}{'precip r':>10}{'r2':>8}{'t2m r':>10}{'r2':>8}")
for nm,bd in {"global":None,"tropics |lat|<=23":np.abs(lat)<=23,
              "midlat 23-50":(np.abs(lat)>23)&(np.abs(lat)<=50),"high >50":np.abs(lat)>50}.items():
    out=[]
    for F,O in ((SPa,TPa),(STa,T2a)):
        m=np.isfinite(F)&np.isfinite(O)
        if bd is not None: m&=bd[None,:]
        r=np.corrcoef(F[m],O[m])[0,1]; out+=[r,r*r]
    print(f"{nm:<22}{out[0]:>10.3f}{out[1]:>8.3f}{out[2]:>10.3f}{out[3]:>8.3f}")
print("saved")
