"""Price the ensembling ceiling from the submissions that already have leaderboard scores.

var(p_i - p_j) == var(e_i - e_j) because the truth cancels, and the published LB scores give
sigma_i, so the error correlation follows without ever seeing the labels.
"""
import pandas as pd, numpy as np
R="d:/Water_4_9_26/Water_Competition/"
LB={"v7":0.653156,"v8":0.657123,"v9":0.651802}
M=pd.DataFrame({k: pd.read_csv(R+f"outputs_{k}/submission_{k}.csv").set_index("ID")["Target"]
                for k in LB})
ks=list(LB); RHO=pd.DataFrame(np.eye(3),index=ks,columns=ks)
for a in range(3):
    for b in range(a+1,3):
        i,j=ks[a],ks[b]
        d=float(np.var(M[i]-M[j]))
        RHO.loc[i,j]=RHO.loc[j,i]=(LB[i]**2+LB[j]**2-d)/(2*LB[i]*LB[j])
print("error correlation between submissions\n"+RHO.round(4).to_string())
C=np.array([[RHO.loc[i,j]*LB[i]*LB[j] for j in ks] for i in ks])
blend=lambda w:(lambda v: float(np.sqrt(v@C@v)))(np.asarray(w,float)/np.sum(w))
print(f"\nv9 alone                          {LB['v9']:.4f}")
print(f"v7+v9 50/50                       {blend([.5,0,.5]):.4f}")
print(f"v7/v8/v9 0.40/0.20/0.40 (built)   {blend([.4,.2,.4]):.4f}")
from scipy.optimize import minimize
w=np.abs(minimize(lambda w: blend(np.abs(w)),[1,1,1],method="Nelder-Mead").x); w/=w.sum()
print(f"best weights {np.round(w,3)}          {blend(w):.4f}")
rho=RHO.values[np.triu_indices(3,1)].mean()
print(f"\nmean pairwise error correlation rho = {rho:.4f}")
for N in (3,5,10,50): print(f"  N={N:>2} at this rho -> {LB['v9']*np.sqrt(rho+(1-rho)/N):.4f}")
print(f"  N=inf          -> {LB['v9']*np.sqrt(rho):.4f}")
print(f"  rho needed for 0.614 with N=10: {((0.614/LB['v9'])**2-0.1)/0.9:.3f}")
