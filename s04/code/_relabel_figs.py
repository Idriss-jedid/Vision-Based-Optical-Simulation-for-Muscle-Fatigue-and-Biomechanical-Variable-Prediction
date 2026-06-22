import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B=r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation\batch"; FIG=os.path.join(B,"report_figs_3d")
# fig3d_progression : renomme Approche A
plt.figure(figsize=(8,4.2))
stg=["3D basic","3D + FE\nenrichi","+ Butterworth","+ Optuna\n(final)"]; vals=[0.842,0.867,0.895,0.904]
plt.bar(stg,vals,color=["#aaa","#7fa","#4c9","#2a7"])
for i,v in enumerate(vals): plt.text(i,v+0.003,"%.3f"%v,ha="center")
plt.axhline(0.952,ls="--",c="r",label="Approach A (.mot + .osim) = 0.952"); plt.ylim(0.8,0.97)
plt.ylabel("R2 moyen (LOSO)"); plt.title("Progression du modele vision-only (3D->biomeca)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig3d_progression.png"),dpi=130); plt.close()
# fig3d_torque
plt.figure(figsize=(7,4)); ts=["baseline","Savitzky-\nGolay","Butterworth","+Optuna"]; tv=[0.790,0.765,0.827,0.839]
plt.bar(ts,tv,color="#d62728")
for i,v in enumerate(tv): plt.text(i,v+0.004,"%.3f"%v,ha="center")
plt.axhline(0.937,ls="--",c="gray",label="Approach A (.mot + .osim)=0.937"); plt.ylim(0.7,0.96)
plt.ylabel("torque R2"); plt.title("Amelioration du torque (vision-only)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig3d_torque.png"),dpi=130); plt.close()
print("figures relabelled")
