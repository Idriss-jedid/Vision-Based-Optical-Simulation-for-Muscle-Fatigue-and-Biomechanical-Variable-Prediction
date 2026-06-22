import json, glob, os, re
import numpy as np
S04 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# .trc RElbow
trc = sorted(f for f in glob.glob(os.path.join(S04,"build2","pose2sim","pose-3d","*.trc")) if "LSTM" not in f)[-1]
L=open(trc).read().splitlines(); names=[m for m in L[3].split("\t") if m.strip()][2:]
D=np.array([[float(x) if x.strip() else np.nan for x in ln.split("\t")] for ln in L[6:] if len(ln.split("\t"))>5])
j=names.index("RElbow"); trc_el=D[:,2+3*j:2+3*j+3]
# Vicon RElbow (joint 15)
J=np.array(json.load(open(os.path.join(S04,"joints3d_25","dumbbell_biceps_curls.json")))["joints3d_25"])
fr=200
print("frame",fr)
print(" .trc  RElbow:", np.round(trc_el[fr],3))
print(" Vicon RElbow:", np.round(J[fr,15],3))
print(" .trc  range x:[%.2f,%.2f] y:[%.2f,%.2f] z:[%.2f,%.2f]"%(np.nanmin(trc_el[:,0]),np.nanmax(trc_el[:,0]),np.nanmin(trc_el[:,1]),np.nanmax(trc_el[:,1]),np.nanmin(trc_el[:,2]),np.nanmax(trc_el[:,2])))
print(" Vicon range x:[%.2f,%.2f] y:[%.2f,%.2f] z:[%.2f,%.2f]"%(J[:,15,0].min(),J[:,15,0].max(),J[:,15,1].min(),J[:,15,1].max(),J[:,15,2].min(),J[:,15,2].max()))
