"""Split v9's error into (a) irreducible, (b) nowcast error, (c) modelling error.

Same harness as 07, same rows, same hyper-parameters. Three extra arms:
  H  baseline + the TRUE TWS anomaly at month t          -> a perfect NOWCAST (floor B, measured)
  I  H + the true target-month ERA5                      -> perfect nowcast AND perfect weather
  J  baseline + the true anomaly at t and at t-1         -> perfect nowcast of state AND tendency
Anything H recovers is nowcast error and is attackable by better features/models.
Anything it does not recover is the one-step innovation plus GRACE noise, and is not.
"""
import numpy as np, pandas as pd, lightgbm as lgb, time, os
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
R="d:/Water_4_9_26/Water_Competition/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); nC=int(z["nC"])
lat=z["lat"]; TWS=z["TWS"]
E={k[2:]:z[k] for k in z.files if k.startswith("e_")}
E["swvl"]=(0.035*E["swvl1"]+0.105*E["swvl2"]+0.36*E["swvl3"]+0.945*E["swvl4"])/1.445
mon=np.array([(m_lo+m-1)%12 for m in range(nM)])
FIT_END=2013*12+7-m_lo
def clim_anom(A,end):
    B=np.full_like(A,np.nan,dtype=np.float32)
    for c in range(12):
        k=np.where(mon==c)[0]; ke=k[k<end]
        B[k]=A[k]-np.nanmean(A[ke],axis=0)
    return B
DEV=clim_anom(TWS,FIT_END); EA={k:clim_anom(v,FIT_END) for k,v in E.items()}
_s=np.load(S+"spei.npz"); SPP={k[2:]:_s[k] for k in _s.files}
OBS=np.isfinite(TWS).sum(1)>200; obsm=np.where(OBS)[0]
VAR=["tp","e","ro","smlt","sd","swvl","t2m","pev","snowc"]
rng=np.random.default_rng(0)
gather=lambda A,m,c: A[np.clip(m,0,nM-1),c]
def make(a,h,c):
    tm=a+h; f={"h":h.astype(np.float32)}
    f["dev"]=gather(DEV,a,c); f["dev1"]=gather(DEV,a-1,c); f["dev2"]=gather(DEV,a-2,c)
    f["d_dev"]=f["dev"]-f["dev1"]; f["d_dev3"]=f["dev"]-gather(DEV,a-3,c)
    ang=2*np.pi*mon[np.clip(tm,0,nM-1)]/12
    f["msin"]=np.sin(ang).astype(np.float32); f["mcos"]=np.cos(ang).astype(np.float32)
    f["abslat"]=np.abs(lat[c]).astype(np.float32)
    for v in VAR:
        A=EA[v]
        for k in (1,2,3): f[f"{v}_l{k}"]=gather(A,tm-k,c)
        acc=np.zeros(len(a),np.float32); n=np.zeros(len(a),np.float32)
        for k in range(1,8):
            vv=np.where(k<=h,gather(A,tm-k,c),np.nan); g=np.isfinite(vv); acc[g]+=vv[g]; n[g]+=1
        f[f"{v}_gap"]=np.where(n>0,acc/np.maximum(n,1),np.nan).astype(np.float32)
    for k,v in SPP.items():
        for q in (1,2,3): f["Q_"+k+str(q)]=gather(v,tm-q,c)
    return f
def add_nowcast(f,a,h,c):
    tm=a+h; f["ORACLE_dev_t"]=gather(DEV,tm-1,c); return f
def add_tendency(f,a,h,c):
    tm=a+h
    f["ORACLE_dev_t1"]=gather(DEV,tm-2,c)
    f["ORACLE_d"]=f["ORACLE_dev_t"]-f["ORACLE_dev_t1"]; return f
def add_weather(f,a,h,c):
    tm=a+h
    for v in VAR: f["F_"+v]=gather(EA[v],tm,c)
    return f
TESTMIX={1:.335,2:.222,3:.166,4:.111,5:.056,6:.055,7:.055}
def sample(rg,per):
    A=[];H=[];C=[]
    ms=[m for m in obsm if rg[0]<=m<rg[1]]
    for h,wt in TESTMIX.items():
        pr=[m for m in ms if (m+h)<nM and OBS[m+h] and m-3>=0]
        if not pr: continue
        k=int(per*wt); A.append(rng.choice(pr,size=k)); H.append(np.full(k,h)); C.append(rng.integers(0,nC,size=k))
    A=np.concatenate(A);H=np.concatenate(H);C=np.concatenate(C)
    y=DEV[A+H,C]; ok=np.isfinite(y)&np.isfinite(DEV[A,C])
    return A[ok],H[ok],C[ok],y[ok]
at,ht,ct,yt=sample((3,2012*12+1-m_lo),300000); av,hv,cv,yv=sample((2012*12+1-m_lo,nM),150000)
print("train",len(yt),"valid",len(yv),flush=True)
P=dict(objective="l2",learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,
       feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=6,seed=1)
def run(name,extra):
    ft=make(at,ht,ct); fv=make(av,hv,cv)
    for fn in extra: fn(ft,at,ht,ct); fn(fv,av,hv,cv)
    cols=list(ft)
    m=lgb.train(P,lgb.Dataset(np.column_stack([ft[c] for c in cols]),yt,feature_name=cols),700)
    r=float(np.sqrt(np.mean((m.predict(np.column_stack([fv[c] for c in cols]))-yv)**2)))
    print(f"  {name:<44} nfeat={len(cols):3d}  RMSE {r:.4f}",flush=True); return r
t0=time.time()
A_=run("A  causal baseline",[])
H_=run("H  + TRUE anomaly at t  (perfect nowcast)",[add_nowcast])
J_=run("J  + true anomaly at t and t-1 (state+tendency)",[add_nowcast,add_tendency])
I_=run("I  H + true target-month ERA5 (perfect everything)",[add_nowcast,add_weather])
print(f"\n{'':<46}{'RMSE':>8}{'MSE':>9}{'share of A''s MSE':>20}")
for nm,r in [("A causal baseline",A_),("H perfect nowcast of the state at t",H_),
             ("J + perfect tendency",J_),("I + perfect target-month weather",I_)]:
    print(f"{nm:<46}{r:>8.4f}{r*r:>9.4f}{100*r*r/(A_*A_):>19.1f}%")
print(f"""
 A - H = {A_*A_-H_*H_:.4f} MSE  ->  the NOWCAST gap: everything better features, a better
         filter, or a better model class could in principle recover ({100*(A_*A_-H_*H_)/(A_*A_):.1f}% of A's error)
 H     = {H_*H_:.4f} MSE  ->  irreducible given the state at t: one-step innovation + GRACE noise
 I     = {I_*I_:.4f} MSE  ->  and knowing the weather too changes it by {I_*I_-H_*H_:+.4f}
 {time.time()-t0:.0f}s""")
