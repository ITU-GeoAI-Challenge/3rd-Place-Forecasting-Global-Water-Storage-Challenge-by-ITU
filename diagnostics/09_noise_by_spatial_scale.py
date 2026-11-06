"""How much of the GRACE target is spatially INCOHERENT noise, and how much survives smoothing?

If the residual v9 cannot explain is spatially white, a predictor that works on a smoothed /
spectral representation of the field removes it for free and the floor drops. If the residual
is spatially coherent, no amount of representation engineering helps and the floor is real.

Estimator: for a field with temporally-white noise on top of an AR-like signal,
    rho1 = s*phi/(s+n),  rho2 = s*phi^2/(s+n)   ->   phi = rho2/rho1,  signal share = rho1^2/rho2
computed per cell over CONSECUTIVE observed months, then area-weighted.
"""
import numpy as np, os
S="d:/Water_4_9_26/Water_Competition/diagnostics/cache/"
z=np.load(S+"panels.npz"); m_lo=int(z["m_lo"]); nM=int(z["nM"]); TWS=z["TWS"]
lat=z["lat"]; lon=z["lon"]
mon=np.array([(m_lo+m-1)%12 for m in range(nM)])
FIT_END=2013*12+7-m_lo
DEV=np.full_like(TWS,np.nan)
for c in range(12):
    k=np.where(mon==c)[0]; ke=k[k<FIT_END]
    DEV[k]=TWS[k]-np.nanmean(TWS[ke],axis=0)
OBS=np.where(np.isfinite(TWS).sum(1)>200)[0]

NLAT,NLON,LAT0,LON0=140,360,-55.5,-179.5
il=np.round(lat-LAT0).astype(int); jl=np.round(lon-LON0).astype(int)
from scipy.ndimage import gaussian_filter
def smooth_panel(A,sig):
    if sig<=0: return A.copy()
    B=np.full_like(A,np.nan)
    for m in range(A.shape[0]):
        if not np.isfinite(A[m]).any(): continue
        G=np.full((NLAT,NLON),np.nan,np.float32); G[il,jl]=A[m]
        w=np.isfinite(G).astype(np.float32); v=np.nan_to_num(G)
        num=gaussian_filter(v*w,sig,mode=("nearest","wrap"))
        den=gaussian_filter(w,  sig,mode=("nearest","wrap"))
        with np.errstate(all="ignore"): R=np.where(den>1e-3,num/den,np.nan)
        B[m]=R[il,jl]
    return B

def shares(A):
    """area-weighted lag-1/lag-2 autocorrelation over consecutive observed months"""
    pairs1=[(a,a+1) for a in OBS if (a+1) in set(OBS.tolist())]
    pairs2=[(a,a+2) for a in OBS if (a+2) in set(OBS.tolist())]
    def rho(pairs):
        num=0.0; d1=0.0; d2=0.0
        for a,b in pairs:
            x,y=A[a],A[b]; g=np.isfinite(x)&np.isfinite(y)
            w=np.cos(np.deg2rad(lat[g]))
            num+=np.sum(w*x[g]*y[g]); d1+=np.sum(w*x[g]**2); d2+=np.sum(w*y[g]**2)
        return num/np.sqrt(d1*d2)
    r1,r2=rho(pairs1),rho(pairs2)
    phi=r2/r1; sig_share=r1*r1/r2
    return r1,r2,phi,min(sig_share,1.0)

V=float(np.nanvar(DEV[OBS]))
print(f"anomaly variance of the observed panel: {V:.4f}\n")
print(f"{'spatial smoothing':<22}{'var kept':>10}{'rho1':>8}{'rho2':>8}{'phi':>8}"
      f"{'signal':>9}{'noise':>8}{'floor B':>9}")
for sig in (0,0.5,0.75,1.0,1.5,2.0,3.0,4.0):
    A=smooth_panel(DEV,sig)
    v=float(np.nanvar(A[OBS]))
    r1,r2,phi,s=shares(A)
    n=1-s
    # floor B on the ORIGINAL scale: a perfect smooth-field forecast still pays the fine-scale
    # variance it threw away, plus the innovation of the part it does model.
    lost=V-v
    fB=np.sqrt(lost + v*n + v*s*(1-phi**2))
    print(f"sigma={sig:<16.2f}{v:>10.4f}{r1:>8.4f}{r2:>8.4f}{phi:>8.4f}{s:>9.3f}{n:>8.3f}{fB:>9.4f}")
print("""
 'signal'  = share of the smoothed field's variance that is temporally coherent
 'noise'   = the rest: temporally white AT THAT SPATIAL SCALE
 'floor B' = best achievable RMSE against the RAW target for a forecaster that models only
             this smoothed field perfectly at time t: (variance thrown away) + (white part)
             + (one-step innovation of the coherent part)""")
