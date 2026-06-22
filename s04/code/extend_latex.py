# pyright: reportMissingImports=false
"""
Etend batch/latex/main.tex (Approche A) en un rapport UNIFIE :
  Part I = existant (.mot+.osim) ; Part II = 3D joints ; Part III = 2D keypoints ;
  Part IV = pipeline (image) + Implicit/Explicit memory + synthese.
Copie les figures 3D/2D/pipeline dans batch/latex/figs/. biomech env.
"""
import os, shutil, re
import pandas as pd

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); LX = os.path.join(B, "latex"); FG = os.path.join(LX, "figs")
os.makedirs(FG, exist_ok=True)

# --- copier figures 3D / 2D / pipeline ---
for src in [os.path.join(B, "report_figs_3d"), os.path.join(B, "report_figs_2d")]:
    if os.path.isdir(src):
        for f in os.listdir(src):
            if f.endswith(".png"): shutil.copy(os.path.join(src, f), os.path.join(FG, f))
pp = os.path.join(B, "report_figs", "pipeline_diagram.png")
if os.path.exists(pp): shutil.copy(pp, os.path.join(FG, "pipeline_diagram.png"))


def esc(s):
    if isinstance(s, float) and pd.isna(s): return ""
    s = str(s)
    if s == "nan": return ""
    for a, b in [("\\", r"\textbackslash "), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")]:
        s = s.replace(a, b)
    return s


def tabular(csv, idx, ndec=3, top=None, cols=None, rename=None):
    d = pd.read_csv(csv, index_col=0)
    if cols: d = d[[c for c in cols if c in d.columns]]
    if top: d = d.head(top)
    rename = rename or {}
    al = "l" + "r" * len(d.columns)
    out = ["\\begin{tabular}{%s}" % al, "\\toprule",
           "\\textbf{%s} & %s \\\\" % (esc(idx), " & ".join("\\textbf{%s}" % esc(rename.get(c, c)) for c in d.columns)), "\\midrule"]
    for ix, row in d.iterrows():
        cells = [esc(ix)]
        for c in d.columns:
            v = row[c]
            cells.append("" if (isinstance(v, float) and pd.isna(v)) else ("%.*f" % (ndec, v) if isinstance(v, float) else esc(v)))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def tbl(csv, idx, cap, **kw):
    return ("\\begin{table}[H]\\centering\n\\begin{adjustbox}{max width=\\textwidth}\n%s\n\\end{adjustbox}\n"
            "\\caption{%s}\n\\end{table}" % (tabular(csv, idx, **kw), cap))


def fig(name, cap, w=0.9):
    if not os.path.exists(os.path.join(FG, name)): return ""
    return ("\\begin{figure}[H]\\centering\n\\includegraphics[width=%.2f\\linewidth]{figs/%s}\n\\caption{%s}\n\\end{figure}" % (w, name, cap))


optcols3 = ["value", "params_n_estimators", "params_num_leaves", "params_learning_rate", "params_subsample", "params_colsample_bytree", "params_min_child_samples"]
ren = {c: c.replace("params_", "").replace("_", "\\_") for c in optcols3}

# ============ PART II : 3D ============
P2 = r"""
\clearpage
\part{Part II --- From 3D joints to biomechanics (vision-only)}
\section{Objective and pipeline (3D)}
Here we drop the \texttt{.mot} and \texttt{.osim} files entirely and predict the 13 biomechanical targets
\emph{directly from the 3 markerless 3D joints} (RShoulder, RElbow, RWrist) produced by Pose2Sim triangulation ---
no Vicon, no OpenSim at inference. The labels for training were generated once by OpenSim (teacher).

\section{From 9 coordinates to features (worked example)}
The raw input is the 3D position (m) of the 3 joints per frame; example (s03, frame 0): RShoulder=(-0.065,1.419,-0.119),
RElbow=(-0.127,1.140,-0.154), RWrist=(-0.177,0.894,-0.094). Raw pixels/coordinates are position-dependent, so we
transform them into invariant quantities:
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt]
  \item Bone vectors: $SE=$ elbow$-$shoulder $=(-0.062,-0.278,-0.035)$, $WE=$ wrist$-$elbow $=(-0.050,-0.246,0.060)$.
  \item Lengths: $|SE|=0.287$\,m, $|WE|=0.258$\,m.
  \item Elbow angle: $q=\arccos(-SE\!\cdot\!WE/(|SE||WE|))=159^\circ$ (arm nearly straight at $t{=}0$).
  \item Elevations vs vertical: $\arccos(SE\!\cdot\!\mathrm{up}/|SE|)$; gravity moment $\propto \sin(\text{forearm-from-vertical})$.
  \item Derivatives $\dot q,\ddot q$ by finite differences \emph{after} a 2\,Hz zero-phase Butterworth filter.
  \item Physics: $grav=g\sin(\alpha)(m_{fa}L/2+2L)$, inertia $=(m_{fa}L^2/3+2L^2)\ddot q$.
  \item Cumulative (fatigue): $\int|\dot q|dt$, $\int|grav|dt$; rolling mean/std; anthropometry (masses).
\end{enumerate}

\paragraph{Butterworth filter.} Low-pass, cutoff $f_c=2$\,Hz, $W_n=f_c/(f_s/2)=0.04$, 2nd order, zero-phase
(\texttt{filtfilt}). It keeps the curl ($\sim$0.5\,Hz) and removes markerless jitter ($\sim$3--6\,Hz); applied
\emph{before} differentiating because differentiation amplifies high-frequency noise. This is the single most
effective lever for the torque (0.77 $\to$ 0.83).

\section{Feature-engineering journey (3D)}
Overall $R^2$: basic 3D $0.842 \to$ enriched FE $0.867 \to$ +Butterworth $0.895 \to$ +Optuna $0.904$;
torque $0.72 \to 0.79 \to 0.83 \to 0.84$.
""" + fig("fig3d_progression.png", "Progression of the 3D vision-only model (dashed: Approach A using .mot+.osim).") + r"""

\section{Torque deep-dive}
$\tau=M(q)\ddot q+C\dot q+G(q)$: the inertial term needs $\ddot q$, the noisiest markerless quantity. Butterworth
smoothing + a torque-specific Optuna raised torque $R^2$ from 0.79 to 0.84 (MAE 1.12 $\to$ 0.83\,N\,m). Explicit
gravity/inertia features were redundant (SHAP: the angle already encodes gravity).
""" + fig("fig3d_torque.png", "Torque improvement (vision-only): baseline -> Savitzky-Golay -> Butterworth -> +Optuna.", 0.7) + r"""

\section{Optuna (3D) --- top trials}
""" + tbl(os.path.join(B, "optuna_3d_trials.csv"), "trial", "Top-8 Optuna trials (3D model).", top=8, cols=optcols3, rename=ren) + r"""

\section{Final 3D model --- multi-metric and per-subject}
""" + tbl(os.path.join(B, "metrics_3d_final.csv"), "Target", "3D model: metrics per target (LOSO).") + "\n" + \
fig("fig3d_metrics.png", "R$^2$ per target (3D vision-only).", 0.7) + "\n" + \
tbl(os.path.join(B, "metrics_3d_per_subject.csv"), "Subject", "3D model: per-subject R$^2$ (LOSO folds).") + "\n" + \
fig("fig3d_per_subject.png", "Per-subject R$^2$ by group (3D).", 0.9) + "\n" + \
fig("fig3d_muscles.png", "3D: per-muscle predicted vs truth over time, held-out subject.", 0.95) + "\n" + \
fig("fig3d_xai.png", "SHAP feature importance (3D).", 0.7) + "\n" + \
fig("fig3d_compare.png", "Model comparison on the 3D task.", 0.8)

# ============ PART III : 2D ============
P3 = r"""
\clearpage
\part{Part III --- From 2D keypoints to biomechanics}
\section{Objective (2D)}
The most direct vision input: the 2D image keypoints (u,v,confidence) of RShoulder/RElbow/RWrist in the 4 cameras,
mapped straight to the 13 targets --- no triangulation, Vicon or OpenSim at inference.

\section{From 2D pixels to features (worked example)}
Example (s04, camera 50591643, frame 100): RShoulder=(518,247), RElbow=(526,322), RWrist=(539,380) px. Raw pixels
are position/perspective-dependent and lose depth (raw-2D model only $R^2=0.63$). We engineer:
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt]
  \item Per-camera 2D bone vectors, normalised lengths, and 2D elbow angle $a2d=\arccos(-se2d\!\cdot\!we2d/(|se2d||we2d|))$.
  \item \textbf{Multi-view fusion (key):} $a_{\text{fused}}=\sum_c \text{conf}_c\,a2d_c/\sum_c \text{conf}_c$ --- a
        confidence-weighted average of the 4 cameras' 2D angles, an \emph{implicit triangulation} that recovers an
        angle close to the true 3D elbow angle.
  \item Butterworth on $a_{\text{fused}}$, then $\dot q,\ddot q$; cumulative $\int|\dot q|dt$; rolling stats; masses.
\end{enumerate}
The fusion alone lifts the model from 0.63 to 0.85 (activations $0.39\to0.86$).

\section{Model selection (2D)}
""" + tbl(os.path.join(B, "research_2d.csv"), "Model", "2D models with feature engineering (LOSO).") + "\n" + \
fig("fig2d_progression.png", "2D progression: raw -> +FE (multi-view fusion) -> +Optuna (dashed: 3D and .mot+.osim).", 0.85) + r"""

\section{Optuna (2D) --- top trials}
""" + tbl(os.path.join(B, "optuna_2d_trials.csv"), "trial", "Top-8 Optuna trials (2D model).", top=8, cols=optcols3, rename=ren) + r"""

\section{Final 2D model --- multi-metric, per-subject, XAI}
""" + tbl(os.path.join(B, "metrics_2d_final.csv"), "Target", "2D model: metrics per target (LOSO).") + "\n" + \
fig("fig2d_metrics.png", "R$^2$ per target (2D).", 0.7) + "\n" + \
tbl(os.path.join(B, "metrics_2d_per_subject.csv"), "Subject", "2D model: per-subject R$^2$ (LOSO folds).") + "\n" + \
fig("fig2d_per_subject.png", "Per-subject R$^2$ by group (2D).", 0.9) + "\n" + \
fig("fig2d_xai.png", "SHAP feature importance (2D): the fused angle and cumulative term dominate.", 0.7) + "\n" + \
fig("fig2d_muscles.png", "2D: per-muscle predicted vs truth over time, held-out subject.", 0.95) + "\n" + \
fig("fig2d_compare.png", "The three approaches compared.", 0.8)

# ============ PART IV : Pipeline + memory + synthesis ============
P4 = r"""
\clearpage
\part{Part IV --- Pipeline overview and synthesis}
\section{The full pipeline (classical descending + AI ascending)}
""" + fig("pipeline_diagram.png", "Classical pipeline (videos -> OpenSim labels, with validation Tests 1-4) and AI surrogates that climb up to replace more of the chain (0.95 -> 0.90 -> 0.86; next: directly from video).", 0.98) + r"""
The classical chain (left, top-down) goes video $\to$ calibration $\to$ 2D pose $\to$ triangulation $\to$ filtering
$\to$ angles $\to$ scaled model $\to$ OpenSim $\to$ labels, each stage validated against Vicon
(2D 8.3\,px; 3D 27.8\,mm/5.5$^\circ$; elbow $r{=}0.993$, MAE 4.2$^\circ$; scaling 0.85--1.05). The AI surrogates
(right) tap the pipeline higher and higher, replacing more of it.

\section{Synthesis: the three approaches}
\begin{table}[H]\centering
\begin{tabular}{llc}
\toprule
\textbf{Approach} & \textbf{Input at inference} & \textbf{mean $R^2$ (LOSO)} \\
\midrule
A --- \texttt{.mot} + \texttt{.osim} & cleaned angles + scaled model & 0.95 \\
3D joints (vision-only) & 3 triangulated 3D joints & 0.90 \\
2D keypoints (4 cams) & raw 2D image keypoints + fusion & 0.86 \\
\bottomrule
\end{tabular}
\caption{The closer the input is to the biomechanics, the higher the accuracy; the gap measures the value of each classical step.}
\end{table}

\section{Implicit vs explicit temporal memory}
Fatigue is the only \emph{cumulative} target: $MF(t)=F\!\int_0^t MA(s)e^{-R(t-s)}ds$. There are two ways to give a
model this history.
\begin{itemize}[leftmargin=1.4em,itemsep=3pt]
  \item \textbf{Implicit memory} --- the model learns the history itself (LSTM, GRU, Transformer). It carries the
        past in its hidden state / attention.
  \item \textbf{Explicit memory} --- we hand the model ready-made cumulative features ($\texttt{elapsed\_time}$,
        $\texttt{cum\_load}$, $\texttt{cum\_work}$, $\texttt{cum\_path}$). The model stays per-frame.
\end{itemize}
\textbf{We used explicit memory, and it is decisive on a small dataset.} Fatigue is essentially
$\text{fatigue}\approx\int \text{load}\,dt$; an LSTM/Transformer has thousands--millions of parameters and would
need \emph{hundreds} of independent subjects to discover that integral from scratch. We have only \textbf{8
subjects}. By computing the integral ourselves and feeding it as a feature, the boosted tree only learns the
trivial monotone map $\texttt{cum\_load}\to\text{fatigue}$ --- far more data-efficient. This is why, across all
experiments, gradient boosting with explicit cumulative features beat the deep sequence models on fatigue, and is
the core reason the whole approach works with so few subjects.

\section{Next step}
The natural continuation is to remove even the keypoint-detection step and learn \emph{directly from the video}
(end-to-end), which would require a much larger, more varied dataset but is the ultimate deployable form.

\section{Overall conclusion}
From ordinary multi-view video, validated markerless kinematics feed an ML surrogate that predicts joint torque,
muscle forces, activations and fatigue without OpenSim at inference. Three input levels were demonstrated ---
\texttt{.mot}+\texttt{.osim} (0.95), 3D joints (0.90), 2D keypoints (0.86) --- all using gradient boosting with
physics-guided (explicit-memory) feature engineering, Optuna tuning and SHAP explainability, under strict
Leave-One-Subject-Out validation.
"""

# ---- splice into main.tex ----
mt = os.path.join(LX, "main.tex"); src = open(mt, encoding="utf-8").read()
# titre unifie
src = src.replace("Approach A (kinematics $\\to$ torque / forces / activations / fatigue) --- Leave-One-Subject-Out, 8 subjects}",
                  "Three input levels: \\texttt{.mot}+\\texttt{.osim} (0.95) $\\cdot$ 3D joints (0.90) $\\cdot$ 2D keypoints (0.86) --- Leave-One-Subject-Out, 8 subjects}")
# Part I divider apres l'abstract
src = src.replace("\\end{abstract}", "\\end{abstract}\n\n\\part{Part I --- Approach A: the \\texttt{.mot}+\\texttt{.osim} pipeline}", 1)
# inserer parts II/III/IV avant \end{document}
src = src.replace("\\end{document}", P2 + "\n" + P3 + "\n" + P4 + "\n\n\\end{document}")
open(mt, "w", encoding="utf-8").write(src)
print("main.tex etendu. lignes:", len(src.splitlines()))
print("figures dans figs/:", len([f for f in os.listdir(FG) if f.endswith('.png')]))
