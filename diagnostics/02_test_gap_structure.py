import pandas as pd, numpy as np
R="d:/Water_4_9_26/Water_Competition/"
te=pd.read_csv(R+"Test.csv",usecols=["ID","time","lat","lon","TWS_t","TWS_t_masked"])
tr=pd.read_csv(R+"Train.csv",usecols=["time","lat","lon","TWS_t"])
te['t']=pd.to_datetime(te.time); te['mk']=te.t.dt.year*12+te.t.dt.month
tr['t']=pd.to_datetime(tr.time); tr['mk']=tr.t.dt.year*12+tr.t.dt.month
g=te.groupby('mk').agg(n=('ID','size'),obs=('TWS_t',lambda s:s.notna().mean()))
g['ym']=[f"{(k-1)//12}-{(k-1)%12+1:02d}" for k in g.index]
print(g.to_string())
obs_tr=sorted(tr.mk.unique()); obs_te=sorted(g.index[g.obs>0.5])
# observation panel available WITHOUT using future test rows = train months (+their targets) and test obs months
print("\ntrain months n=",len(obs_tr),"last:",[f"{(k-1)//12}-{(k-1)%12+1:02d}" for k in obs_tr[-4:]])
tgt=set(k+1 for k in obs_tr)                      # train targets are observed TWS too
known=sorted(set(obs_tr)|tgt|set(obs_te))
print("observed TWS months (train+train targets+test obs):",len(known))
print("test-era observed:",[f"{(k-1)//12}-{(k-1)%12+1:02d}" for k in known if k>=obs_te[0]-2])

# per test row: target month = mk+1 ; nearest observed BEFORE (<= mk) and AFTER (>= mk+2)
K=np.array(known)
rows=[]
for mk,n in zip(g.index,g.n):
    tmm=mk+1
    prev=K[K<=mk].max()
    nxt=K[K>=tmm]
    nxt=nxt.min() if len(nxt) else None
    rows.append((f"{(mk-1)//12}-{(mk-1)%12+1:02d}",n,tmm-prev,(nxt-tmm) if nxt is not None else -1))
D=pd.DataFrame(rows,columns=["test_month","n","h_back","h_fwd"])
print("\n",D.to_string(index=False))
print("\nweighted mean h_back:",np.average(D.h_back,weights=D.n))
print("share h_back>=4:",D.n[D.h_back>=4].sum()/D.n.sum())
print("share with a FUTURE anchor at all:",D.n[D.h_fwd>=0].sum()/D.n.sum())
D.to_csv("d:/Water_4_9_26/Water_Competition/_gapstruct.csv",index=False)
