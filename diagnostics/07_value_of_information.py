import os
os.makedirs("d:/Water_4_9_26/Water_Competition/diagnostics/cache/",exist_ok=True)
import numpy as np, pandas as pd, lightgbm as lgb, time, os
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
R="d:/Water_4_9_26/Water_Competition/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); nC=int(z["nC"])
lat=z["lat"]; TWS=z["TWS"]
E={k[2:]:z[k] for k in z.files if k.startswith("e_")}
E["swvl"]=(0.035*E["swvl1"]+0.105*E["swvl2"]+0.36*E["swvl3"]+0.945*E["swvl4"])/1.445
mon=np.array([(m_lo+m-1)%12 for m in range(nM)])
FIT_END=2013*12+7-m_lo          # climatology/anomaly fitted on months before this
def clim_anom(A,end):
    B=np.full_like(A,np.nan,dtype=np.float32)
    for c in range(12):
        k=np.where(mon==c)[0]; ke=k[k<end]
        mu=np.nanmean(A[ke],axis=0); B[k]=A[k]-mu
    return B
DEV=clim_anom(TWS,FIT_END)
EA={k:clim_anom(v,FIT_END) for k,v in E.items()}
SE=np.load(S+"seas5.npz") if os.path.exists(S+"seas5.npz") else None
if SE is not None:
    SEA={k:clim_anom(SE[k],FIT_END) for k in ("s_tp","s_t2m")}
SPP=None
try:
    _s=np.load(S+"spei.npz"); SPP={k[2:]:_s[k] for k in _s.files}
    print("provided-column panels:",list(SPP))
except Exception as e: print("spei skip",e)
GD=None
try:
    g=np.load(R+"gdo/gdo_1deg_monthly.npz")
    gi=np.round((float(g["lat"][0])-lat)).astype(int); gj=np.round((z["lon"]-float(g["lon"][0]))).astype(int)
    mk=g["months"].astype(int); GD={}
    for k in ("spg03","spg06","fapan"):
        if k not in g.files: continue
        a=np.full((nM,nC),np.nan,np.float32)
        arr=g[k]                       # decompress ONCE, not once per month
        for q,M in enumerate(mk):
            m=int(M)-m_lo
            if 0<=m<nM: a[m]=arr[q][gi,gj]
        del arr
        GD["g_"+k]=a
    print("GDO panels:",{k:round(float(np.isfinite(v).mean()),3) for k,v in GD.items()})
except Exception as e: print("GDO skip",type(e).__name__,e)

OBS=np.isfinite(TWS).sum(1)>200
obsm=np.where(OBS)[0]
print("observed months:",len(obsm))
VAR=["tp","e","ro","smlt","sd","swvl","t2m","pev","snowc"]
rng=np.random.default_rng(0)
def gather(A,m,c):
    m=np.clip(m,0,nM-1); return A[m,c]
def make(anchors,hs,cells):
    tm=anchors+hs
    f={}
    f["h"]=hs.astype(np.float32)
    f["dev"]=gather(DEV,anchors,cells); f["dev1"]=gather(DEV,anchors-1,cells)
    f["dev2"]=gather(DEV,anchors-2,cells)
    f["d_dev"]=f["dev"]-f["dev1"]; f["d_dev3"]=f["dev"]-gather(DEV,anchors-3,cells)
    ang=2*np.pi*mon[np.clip(tm,0,nM-1)]/12
    f["msin"]=np.sin(ang).astype(np.float32); f["mcos"]=np.cos(ang).astype(np.float32)
    f["abslat"]=np.abs(lat[cells]).astype(np.float32)
    for v in VAR:
        A=EA[v]
        for k in (1,2,3): f[f"{v}_l{k}"]=gather(A,tm-k,cells)
        acc=np.zeros(len(anchors),np.float32); n=np.zeros(len(anchors),np.float32)
        for k in range(1,8):
            ok=(k<=hs); vv=np.where(ok,gather(A,tm-k,cells),np.nan)
            g=np.isfinite(vv); acc[g]+=vv[g]; n[g]+=1
        f[f"{v}_gap"]=np.where(n>0,acc/np.maximum(n,1),np.nan).astype(np.float32)
    return f
def add_foresight(f,anchors,hs,cells):
    tm=anchors+hs
    for v in VAR: f[f"F_{v}"]=gather(EA[v],tm,cells)
    return f
def add_seas(f,anchors,hs,cells):
    tm=anchors+hs
    for k in ("s_tp","s_t2m"):
        f["S_"+k]=gather(SEA[k],tm,cells); f["S_"+k+"_p"]=gather(SEA[k],tm-1,cells)
    return f
def add_prov(f,anchors,hs,cells):
    """the provided SPEI / soil-moisture read from the row dated t+1 (the target month).
       Measured above: those values actually describe month t-2..t-1, so they are causal."""
    tm=anchors+hs
    for k,v in SPP.items():
        f["P_"+k]=gather(v,tm,cells)
        f["P_"+k+"_d"]=f["P_"+k]-gather(v,tm-1,cells)
    return f
def add_prov_causal(f,anchors,hs,cells):
    """the same columns read only up to the row dated t -- what v9 already does."""
    tm=anchors+hs
    for k,v in SPP.items():
        for q in (1,2,3): f["Q_"+k+str(q)]=gather(v,tm-q,cells)
    return f
def add_gdo(f,anchors,hs,cells):
    tm=anchors+hs
    for k,v in GD.items():
        f["G_"+k]=gather(v,tm-1,cells); f["G_"+k+"_d"]=f["G_"+k]-gather(v,anchors,cells)
    return f

TESTMIX={1:.335,2:.222,3:.166,4:.111,5:.056,6:.055,7:.055}
def sample(mrange,per=90000):
    A=[];H=[];C=[]
    ms=[m for m in obsm if mrange[0]<=m<mrange[1]]
    for h,wt in TESTMIX.items():
        pairs=[m for m in ms if (m+h)<nM and OBS[m+h] and m-3>=0]
        if not pairs: continue
        k=int(per*wt); 
        mm=rng.choice(pairs,size=k)
        cc=rng.integers(0,nC,size=k)
        A.append(mm);H.append(np.full(k,h));C.append(cc)
    A=np.concatenate(A);H=np.concatenate(H);C=np.concatenate(C)
    y=DEV[A+H,C]; ok=np.isfinite(y)&np.isfinite(DEV[A,C])
    return A[ok],H[ok],C[ok],y[ok]

TRN=(3,2012*12+1-m_lo); VAL=(2012*12+1-m_lo,nM)
print("FIT_END",FIT_END,"TRN",TRN,"VAL",VAL)
at,ht,ct,yt=sample(TRN,300000); av,hv,cv,yv=sample(VAL,150000)
print("train",len(yt),"valid",len(yv))
def run(name,extra):
    ft=make(at,ht,ct); fv=make(av,hv,cv)
    for fn in extra: fn(ft,at,ht,ct); fn(fv,av,hv,cv)
    cols=list(ft.keys())
    Xt=np.column_stack([ft[c] for c in cols]); Xv=np.column_stack([fv[c] for c in cols])
    p=dict(objective="l2",learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,
           feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=6,seed=1)
    m=lgb.train(p,lgb.Dataset(Xt,yt,feature_name=cols),num_boost_round=700)
    r=float(np.sqrt(np.mean((m.predict(Xv)-yv)**2)))
    print(f"  {name:<38} nfeat={len(cols):3d}  RMSE {r:.4f}")
    return r,m,cols
t0=time.time()
base,_,_=run("A causal (TWS + ERA5 up to t)",[add_prov_causal] if SPP else [])
res={}
if SPP is not None: res["+provided SPEI/SM from the t+1 row"]=run("A2 + provided cols at the t+1 row",[add_prov_causal,add_prov])[0]
EX=[add_prov_causal] if SPP else []
if SE is not None: res["+SEAS5 forecast of target month"]=run("B  + SEAS5 (realistic forecast)",EX+[add_seas])[0]
if GD is not None: res["+GDO SPI/fAPAR at t"]=run("C  + GDO SPI3/SPI6/fAPAR",EX+[add_gdo])[0]
res["+PERFECT target-month ERA5"]=run("D  + PERFECT target-month ERA5 (ceiling)",EX+[add_foresight])[0]
if SE is not None and GD is not None:
    res["+SEAS5 +GDO"]=run("E  + SEAS5 + GDO",EX+[add_seas,add_gdo])[0]
    if SPP is not None:
        res["EVERYTHING legitimate"]=run("F  + SEAS5 + GDO + provided-at-t+1",EX+[add_seas,add_gdo,add_prov])[0]
        res["EVERYTHING + perfect weather"]=run("G  F + PERFECT target-month ERA5",EX+[add_seas,add_gdo,add_prov,add_foresight])[0]
print(f"\nbaseline {base:.4f}")
for k,v in res.items(): print(f"  {k:<34} {v:.4f}   delta {v-base:+.4f}")
print(f"\n{time.time()-t0:.0f}s")
