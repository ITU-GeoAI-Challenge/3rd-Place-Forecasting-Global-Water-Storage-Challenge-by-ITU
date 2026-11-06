import pandas as pd, numpy as np
D=pd.read_csv("d:/Water_4_9_26/Water_Competition/_gapstruct.csv")
V=1.149328589439392; w=D.n.values/D.n.sum(); b=D.h_fwd.values
print("Floor B (perfect nowcast of the state at t) vs the same PLUS a future GRACE anchor\n"
      "read out of Test.csv itself -- i.e. the best a NON-causal interpolator could do.\n")
print(f"{'param set':<28}{'causal':>10}{'+future anchor':>16}{'gain':>8}")
for nm,nu,phi in [("pooled  nu13.4 phi.877",0.134,0.877),("late    nu23.3 phi.892",0.233,0.892),
                  ("2-comp  nu16.7 phi.9165",0.1667,0.9165)]:
    s=1-nu
    one=np.full(len(w),V*nu+V*s*(1-phi**2))
    two=one.copy(); m=b>0
    two[m]=V*nu+V*s*(1-phi**2)*(1-phi**(2*b[m]))/(1-phi**(2*(1+b[m])))
    o=np.sqrt((w*one).sum()); t=np.sqrt((w*two).sum())
    print(f"{nm:<28}{o:>10.4f}{t:>16.4f}{t-o:>8.4f}")
print("\nand if the anchors themselves were noise-free (noise only in the SCORED month):")
for nm,nu,phi in [("pooled",0.134,0.877),("2-comp",0.1667,0.9165)]:
    s=1-nu
    two=np.full(len(w),V*nu+V*s*(1-phi**2)); m=b>0
    two[m]=V*nu+V*s*(1-phi**2)*(1-phi**(2*b[m]))/(1-phi**(2*(1+b[m])))
    print(f"  {nm}: {np.sqrt((w*two).sum()):.4f}")
