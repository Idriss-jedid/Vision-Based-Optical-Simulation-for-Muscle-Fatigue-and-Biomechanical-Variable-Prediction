# pyright: reportMissingImports=false
"""
Version LaTeX/Overleaf (etendue + tables non debordantes via adjustbox) ->
batch/latex/main.tex (+ figs/). Dossier autonome a importer dans Overleaf. biomech env.
"""
import os, shutil
import pandas as pd

ROOT = r"C:\Users\21652\Downloads\OpenSimOverView\Vision-Based Optical Simulation"
B = os.path.join(ROOT, "batch"); FIG = os.path.join(B, "report_figs")
LX = os.path.join(B, "latex"); FG = os.path.join(LX, "figs"); os.makedirs(FG, exist_ok=True)

FIGS = ["fig_fe_gain", "fig_tabular", "fig_deep", "fig_optuna_history", "fig_optuna_importance",
        "fig_xai_heatmap", "xai_shap_fatigue", "fig_metrics_bars", "fig_per_subject",
        "fig_muscles_s11", "fig_muscles_s04", "fig_muscles_s08", "fig_torque", "fig_parity", "fig_ts_fatigue"]
for f in FIGS:
    s = os.path.join(FIG, f + ".png")
    if os.path.exists(s): shutil.copy(s, os.path.join(FG, f + ".png"))


def esc(s):
    if isinstance(s, float) and pd.isna(s): return ""
    s = str(s)
    if s == "nan": return ""
    for a, b in [("\\", r"\textbackslash "), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")]:
        s = s.replace(a, b)
    return s


def tabular(csv, idx_name, ndec=3, top=None, cols=None, rename=None):
    d = pd.read_csv(csv, index_col=0)
    if cols: d = d[cols]
    if top: d = d.head(top)
    rename = rename or {}
    al = "l" + "r" * len(d.columns)
    out = ["\\begin{tabular}{%s}" % al, "\\toprule",
           "\\textbf{%s} & %s \\\\" % (esc(idx_name), " & ".join("\\textbf{%s}" % esc(rename.get(c, c)) for c in d.columns)),
           "\\midrule"]
    for ix, row in d.iterrows():
        cells = [esc(ix)]
        for c in d.columns:
            v = row[c]
            if isinstance(v, float) and pd.isna(v): cells.append("")
            elif isinstance(v, float): cells.append("%.*f" % (ndec, v))
            else: cells.append(esc(v))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def tbl(body, cap):
    return ("\\begin{table}[H]\\centering\n\\begin{adjustbox}{max width=\\textwidth}\n%s\n"
            "\\end{adjustbox}\n\\caption{%s}\n\\end{table}" % (body, cap))


def fig(name, cap, w=0.92):
    return ("\\begin{figure}[H]\\centering\n\\includegraphics[width=%.2f\\linewidth]{figs/%s.png}\n"
            "\\caption{%s}\n\\end{figure}" % (w, name, cap))


FEAT = [
    ("q\\_sh", "Shoulder elevation angle [deg]", "Gravity term $G(q)$: arm orientation vs gravity. Arm horizontal $\\to$ max gravitational moment."),
    ("q\\_el", "Elbow flexion angle [deg]", "Forearm orientation and muscle moment arms $r(q)$; spans the curl ROM (0--128 deg)."),
    ("qd\\_sh", "$\\dot q_{sh}=dq_{sh}/dt$ [deg/s]", "Coriolis/centrifugal term $C(q,\\dot q)\\dot q$."),
    ("qd\\_el", "$\\dot q_{el}$ [deg/s]", "Coriolis term; muscle force--velocity relation."),
    ("qdd\\_sh", "$\\ddot q_{sh}$ [deg/s$^2$]", "Inertial term $M(q)\\ddot q$. At reversals $\\ddot q$ is large $\\to$ large inertial torque."),
    ("qdd\\_el", "$\\ddot q_{el}$ [deg/s$^2$]", "Inertial term (elbow)."),
    ("time", "Elapsed time [s]", "Weak cumulative proxy (only base feature carrying history)."),
    ("humerus\\_mass", "Upper-arm mass [kg]", "Scales $M$ and $G$; a heavier arm gives more torque at the same pose. Subject-aware."),
    ("forearm\\_mass", "Forearm mass [kg]", "Dominates the elbow gravity torque (forearm $+$ 2\\,kg load)."),
    ("humerus\\_len", "Humerus length [m]", "Moment arms and inertia scale with segment length."),
    ("forearm\\_len", "Forearm length [m]", "Lever arm of forearm gravity at the elbow."),
    ("sin/cos\\_qel", "$\\sin/\\cos(q_{el})$", "Moment and moment arms are trigonometric in the angle; smooth monotone transforms ease tree splits."),
    ("sin/cos\\_qsh", "$\\sin/\\cos(q_{sh})$", "Same, for shoulder orientation."),
    ("abs\\_qd\\_el", "$|\\dot q_{el}|$", "Speed magnitude (direction-agnostic effort); feeds the cumulative path."),
    ("abs\\_qdd\\_el", "$|\\ddot q_{el}|$", "Inertial-demand magnitude."),
    ("qd\\_el2", "$\\dot q_{el}^2$", "Kinetic-energy proxy ($KE\\sim v^2$): non-linear velocity effect."),
    ("grav\\_load", "$(m_{fa}{+}2)L_{fa}\\sin(q_{sh}{+}q_{el})$", "Static gravitational moment at the elbow. Forearm vertical $\\to\\approx0$; horizontal $\\to$ max."),
    ("qel\\_x\\_fmass", "$q_{el}\\cdot m_{fa}$", "Interaction: posture $\\times$ body size."),
    ("cum\\_path\\_el", "$\\int|\\dot q_{el}|\\,dt$ (/subject)", "Total angular path; proxy of accumulated work/activation $\\to$ fatigue driver; rises over reps."),
    ("cum\\_grav\\_imp", "$\\int|grav\\_load|\\,dt$", "Cumulative gravitational impulse $=$ the dose driving 3CC fatigue. \\textbf{THE} fatigue feature (top SHAP)."),
]


def feat_longtable():
    out = ["{\\small\\begin{longtable}{p{2.5cm} p{4.2cm} p{8.3cm}}", "\\toprule",
           "\\textbf{Feature} & \\textbf{Definition} & \\textbf{Rationale / role (example)} \\\\", "\\midrule", "\\endhead"]
    for a, b, c in FEAT:
        out.append("%s & %s & %s \\\\\\addlinespace" % (a, b, c))
    out += ["\\bottomrule", "\\end{longtable}}"]
    return "\n".join(out)


ot = pd.read_csv(os.path.join(B, "optuna_trials.csv")).sort_values("value", ascending=False)
pc = [c for c in ot.columns if c.startswith("params_")]
ren_opt = {c: c.replace("params_", "").replace("_", "\\_") for c in pc}

DOC = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,longtable,amsmath,float,caption,hyperref,xcolor,adjustbox,enumitem}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{4pt}
\title{\textbf{Machine-Learning Prediction of Internal Biomechanics and Muscle Fatigue from Markerless Vision}\\[4pt]
\large A surrogate model replacing the OpenSim ID\,+\,SO\,+\,3CC pipeline at inference\\
Approach A (kinematics $\to$ torque / forces / activations / fatigue) --- Leave-One-Subject-Out, 8 subjects}
\author{}\date{}
\begin{document}\maketitle
\tableofcontents
\newpage

\begin{abstract}
We build a surrogate machine-learning model that predicts the internal biomechanics of an elbow dumbbell-curl ---
joint torque, individual muscle forces and activations, and muscle fatigue --- directly from markerless kinematics,
thereby replacing the OpenSim biomechanical pipeline (Inverse Dynamics, Static Optimization, and the
3-Compartment-Controller fatigue model) at inference time. Using 8 subjects ($\sim$12{,}640 frames) and a strict
Leave-One-Subject-Out (LOSO) protocol, the retained model --- a gradient-boosted tree ensemble (LightGBM) with
engineered physics-based features and Bayesian hyper-parameter tuning (Optuna) --- reaches a mean $R^2$ of $0.95$
on completely unseen subjects: torque $R^2{=}0.92$, activations $0.96$--$0.97$, forces $0.96$--$0.98$, fatigue
$0.89$--$0.96$. We detail the data splitting, the input/output design and a full feature dictionary, the
feature-engineering rationale and its measured impact, the model benchmark, the Optuna optimisation (with the full
top-$K$ table), the explainability analysis (SHAP), and a multi-metric, time-resolved, per-muscle comparison
against the ground truth.
\end{abstract}

\section{Introduction and objective}
OpenSim computes internal biomechanics through physics-based inverse methods that require a calibrated
musculoskeletal model and are computationally heavy. The goal is to \emph{learn} the mapping from observable
kinematics to these internal quantities on data generated by OpenSim, and then to use the learned model alone at
inference --- enabling fast, real-time-capable estimation without running OpenSim.
\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
  \item \textbf{Approach A (retained).} Input $=$ joint kinematics (angles and derivatives) $+$ subject
        anthropometry; output $=$ the 13 biomechanical variables. Inputs are exactly the physical quantities that
        determine the targets, making the learning problem well-posed and data-efficient.
  \item \textbf{Approach B (discarded).} Input $=$ raw 2D vision key-points; less physically grounded, removed.
\end{itemize}

\section{Pipeline overview}
The end-to-end chain is:
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt]
  \item Multi-camera video (4 cameras, 50\,fps) $\to$ Pose2Sim $\to$ triangulated 3D points (\texttt{.trc}).
  \item 3D points $\to$ joint angles (shoulder, elbow) $\to$ \texttt{arm26} motion (\texttt{.mot}).
  \item Per-subject scaled \texttt{arm26} model $\to$ OpenSim ID $+$ SO $+$ 3CC $\to$ \textbf{labels}.
  \item Kinematics $+$ anthropometry ($X$) and labels ($Y$) $\to$ ML dataset $\to$ training $+$ LOSO validation.
\end{enumerate}
A calibration finding underpins the vision stage: Pose2Sim expects the OpenCV translation $\mathbf{t}=-R\,T$
(not the camera centre $T$); this correction raised the triangulation correlation from $\sim$0.86 to 0.999.

\section{Data and train/test splitting}
The dataset (\texttt{ml\_dataset\_A.csv}) contains 12{,}640 frames from 8 subjects performing 5-repetition dumbbell
biceps curls ($\sim$14--21\,s each, 2\,kg load). Each row is one time frame. Labels are produced by OpenSim on a
per-subject scaled \texttt{arm26} model (2 DOF, 7 muscles, 4 elbow flexors).
\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
  \item Frames per subject: s03=1413, s04=1759, s05=1385, s07=1529, s08=1349, s09=1435, s10=1625, s11=2145.
\end{itemize}
\paragraph{Leave-One-Subject-Out (LOSO) cross-validation.} We do \emph{not} split frames randomly: consecutive
frames of a curl are strongly correlated, so a random split would leak information between train and test and
inflate the score. We split \textbf{by subject}: train on 7 subjects, test on the unseen 8th, repeated so that each
subject is the test set exactly once (8 folds). Reported metrics are averaged over the 8 folds. LOSO measures the
only quantity that matters for deployment --- generalisation to a \emph{new person}.

\section{Generating the labels (OpenSim) --- theory}
\textbf{Inverse Dynamics (ID) --- joint torque:}
\begin{equation}
\tau = \underbrace{M(q)\,\ddot q}_{\text{inertia}} + \underbrace{C(q,\dot q)\,\dot q}_{\text{Coriolis}}
       + \underbrace{G(q)}_{\text{gravity}} - \tau_{\text{ext}} .
\end{equation}
\textbf{Static Optimization (SO) --- muscle forces and activations:}
\begin{equation}
\min_{a}\ \sum_m a_m^2 \qquad \text{s.t.}\qquad \sum_m r_m(q)\,F_m = \tau,\quad 0\le a_m\le 1 .
\end{equation}
\textbf{3-Compartment Controller (3CC) --- fatigue:}
\begin{equation}
\frac{dMF}{dt} = F\,MA(t) - R\,MF(t)
\qquad\Longrightarrow\qquad
MF(t) = F\!\int_0^t MA(s)\,e^{-R(t-s)}\,ds .
\end{equation}
Torque and SO outputs are \emph{instantaneous} (memoryless); fatigue $MF$ is a time-\emph{integral} of activation,
i.e. a \emph{cumulative} quantity --- the central fact motivating the cumulative features (Sec.~\ref{sec:fe}).

\section{Inputs (features) and outputs (labels)}
\textbf{Outputs (13):} torque \texttt{elbow\_moment} [N\,m] (ID); 4 activations \texttt{act\_*} [0--1] (SO);
4 forces \texttt{frc\_*} [N] (SO); 4 fatigue \texttt{MF\_*} [\% capacity] (3CC). Muscles: BIClong, BICshort, BRA,
BRD\_hand.\\
\textbf{Inputs (11 base):} kinematics $q_{sh},q_{el},\dot q_{sh},\dot q_{el},\ddot q_{sh},\ddot q_{el}$, time; and
anthropometry (constant per subject) humerus/forearm mass and length.
\paragraph{A single unified model.} All 13 targets are predicted by \emph{one} model with one shared configuration
(trial \#11) --- including fatigue, handled by the same per-frame model thanks to the cumulative features.
Technically LightGBM is single-output, so the multi-output wrapper trains one identical LightGBM per target; from a
usage standpoint it is a single model mapping 22 inputs to 13 outputs. The deep sequence models
(Sec.~\ref{sec:ts}) are a separate experiment, not part of this model.

\subsection{Feature dictionary: definition, rationale and examples}
Features are \emph{not} arbitrary: each maps to a term of the physics (Eqs.~1--3). Torque needs $q,\dot q,\ddot q$
and the masses/lengths; SO depends on the same state via the torque and moment arms $r(q)$; fatigue is an integral,
hence the cumulative features.
""" + feat_longtable() + r"""

\section{Feature engineering: why, how, impact}\label{sec:fe}
The 11 base features suffice for the instantaneous targets but \emph{not} for fatigue (cumulative): at the same
pose a muscle is more fatigued late than early, which a per-frame model cannot tell from instantaneous kinematics.
We therefore added features that expose the geometry/gravity non-linearities \emph{and} encode the history:
\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
  \item \textbf{Geometry/gravity:} $\sin/\cos(q_{el})$, $\sin/\cos(q_{sh})$, and
        $grav\_load=(m_{fa}{+}2)L_{fa}\sin(q_{sh}{+}q_{el})$.
  \item \textbf{Kinetics:} $|\dot q_{el}|$, $|\ddot q_{el}|$, $\dot q_{el}^2$.
  \item \textbf{Interaction:} $q_{el}\cdot m_{fa}$.
  \item \textbf{Cumulative (key for fatigue):} $cum\_path\_el=\int|\dot q_{el}|dt$,
        $cum\_grav\_imp=\int|grav\_load|dt$.
\end{itemize}
The mis-counted \texttt{rep} feature (6 vs the true 5 repetitions) was removed. \textbf{Impact} (LOSO, LightGBM):
overall $R^2$ $0.918\to0.951$, and fatigue $0.842\to0.929$; Bayesian tuning then raised fatigue to $0.936$.
""" + fig("fig_fe_gain", "Effect of feature engineering and Optuna tuning per target group (LOSO).") + r"""

\section{Model selection: benchmark}
\subsection{Tabular models}
All families are evaluated under identical LOSO conditions.
""" + tbl(tabular(os.path.join(B, "bench_tabular.csv"), "Model"), "Tabular models (8 subjects, LOSO).") + "\n" + \
fig("fig_tabular", "Mean $R^2$ of tabular models; gradient boosting dominates.", 0.8) + r"""

\subsection{Deep-learning models (per-frame and sequential)}
Sequential models use a causal 32-frame window.
\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
  \item \textbf{ANN}: per-frame MLP (128--64--13), no temporal window.
  \item \textbf{1D-CNN}: two 1D-conv layers over the window $+$ global average pooling.
  \item \textbf{LSTM}: 2-layer LSTM (hidden 96); last hidden state $\to$ output.
  \item \textbf{CNN+LSTM}: a 1D-conv front-end feeding the LSTM.
  \item \textbf{Transformer}: a standard Transformer \emph{encoder} over the window --- each frame is linearly
        embedded as a token with a learned positional encoding, 2 encoder layers, 4 attention heads, mean-pooled.
        This is a Time-Series Transformer (TST), \emph{not} PatchTST; the patch-based PatchTST and a separate TST
        encoder are evaluated for fatigue in Sec.~\ref{sec:ts}.
\end{itemize}
""" + tbl(tabular(os.path.join(B, "bench_deep.csv"), "Model"), "Deep models (8 subjects, LOSO).") + "\n" + \
fig("fig_deep", "Deep models (ANN, 1D-CNN, LSTM, CNN+LSTM, Transformer-encoder/TST).", 0.8) + r"""
On this dataset size (8 subjects), gradient-boosted trees clearly outperform both classical and deep models;
LightGBM with engineered features is retained.

\section{Hyper-parameter optimisation with Optuna}
\paragraph{What is tuned.} The optimisation tunes the LightGBM that uses the \textbf{22 engineered} features
(Sec.~\ref{sec:fe}), \emph{not} the 11 base features (which gave poor fatigue, $0.842$). Feature engineering is
applied first; Optuna then searches the hyper-parameters of that 22-feature model --- the feature set is fixed
across trials, only the hyper-parameters change.
\paragraph{How it works.} Optuna performs Bayesian optimisation with a Tree-structured Parzen Estimator (TPE):
each trial is one hyper-parameter set, scored by the LOSO objective
$0.5\,R^2_{\text{fatigue}}+0.5\,R^2_{\text{overall}}$ (to emphasise the hardest target). TPE samples promising
regions of the search space; a Hyperband pruner stops weak trials early (a BOHB-like scheme). 40 trials were run.
Optuna returns, for \emph{every} trial, the score and the exact hyper-parameters; the top-10:
""" + tbl(tabular(os.path.join(B, "optuna_trials.csv"), "trial \\#", ndec=3, top=10,
                  cols=["value"] + pc, rename=ren_opt), "Top-10 Optuna trials with hyper-parameters.") + r"""
\paragraph{Interpretation of trial \#11.} The retained configuration is dominated by \textbf{strong regularisation}
--- \texttt{num\_leaves}=15 (shallow trees), \texttt{reg\_lambda}=8.19 (high $L_2$), \texttt{subsample}=0.605
(bagging), \texttt{min\_child\_samples}=24 --- combined into many trees (\texttt{n\_estimators}=975,
\texttt{learning\_rate}=0.196). This ``many weak, well-regularised learners'' recipe is exactly what generalises
from few subjects.
\paragraph{A compromise, not a per-target optimum.} Trial \#11 optimises a \emph{combined} objective weighted
toward fatigue; it is the best single shared configuration but not optimal for each target alone (untuned torque
$0.942$ vs $0.937$ tuned, traded for fatigue $0.929\to0.936$).
""" + fig("fig_optuna_history", "Optuna optimisation history (objective vs trial).", 0.78) + "\n" + \
fig("fig_optuna_importance", "Hyper-parameter importance (Optuna).", 0.78) + r"""

\section{Explainable AI (XAI)}
\textbf{Why \& how.} A surrogate replacing a physics model must be trusted; we use SHAP (TreeExplainer, exact for
tree ensembles) and permutation importance as a cross-check.
\paragraph{Honest role.} XAI was used for \textbf{interpretation/validation}, not as a performance booster:
XAI-guided feature selection did \emph{not} improve the score (full 22 $=0.952$; top-12 $=0.951$; top-8 $=0.945$),
so the full set was kept. Its value is trust --- it confirms the cumulative features drive fatigue and that gravity
load and inertia drive torque, matching the physics.
""" + tbl(tabular(os.path.join(B, "xai_importance.csv"), "Feature", ndec=3), "SHAP mean$|$value$|$ per feature and target.") + "\n" + \
fig("fig_xai_heatmap", "SHAP importance heatmap (top features $\\times$ targets).", 0.72) + "\n" + \
fig("xai_shap_fatigue", "Most important features for fatigue (SHAP).", 0.68) + r"""
\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
  \item \textbf{Torque}: $grav\_load$ (0.30), $\cos(q_{el})$ (0.27), $\ddot q_{el}$ (0.24) --- gravity $+$ posture $+$ inertia.
  \item \textbf{Activations/forces}: $q_{el}$ (moment arm) and $\dot q_{el}$.
  \item \textbf{Fatigue}: $cum\_path\_el$ (0.48) and $cum\_grav\_imp$ (0.21) --- the cumulative features, as expected.
\end{itemize}

\section{Final model: multi-metric evaluation}
Retained: LightGBM (multi-output), 22 engineered features, Optuna trial \#11, LOSO on 8 subjects. Five metrics per
target: $R^2$, RMSE, MAE, Pearson $r$, NRMSE (\%).
""" + tbl(tabular(os.path.join(B, "metrics_final.csv"), "Target", ndec=3), "Final model metrics per target (LOSO).") + "\n" + \
fig("fig_metrics_bars", "$R^2$ and normalised RMSE per target.", 0.92) + r"""
\paragraph{Per-subject generalisation (held-out = the 8 CV folds).}
""" + tbl(tabular(os.path.join(B, "metrics_per_subject.csv"), "Subject", ndec=3), "Per-subject $R^2$ across the 8 LOSO folds.") + "\n" + \
fig("fig_per_subject", "Per-subject $R^2$ by target group (LOSO).", 0.92) + r"""

\section{Time-resolved comparison against ground truth}
To make the muscle identity explicit, the comparison is shown \textbf{per muscle}: for each flexor (BIClong,
BICshort, BRA, BRD\_hand) we plot its activation, force and fatigue over time (predicted vs ground truth). Torque,
a joint-level quantity, is shown separately.
""" + fig("fig_muscles_s11", "Per-muscle predicted vs truth over time, subject s11 (held-out). Rows = muscles; columns = activation/force/fatigue.") + "\n" + \
fig("fig_muscles_s04", "Per-muscle view, subject s04 (held-out).") + "\n" + \
fig("fig_muscles_s08", "Per-muscle view, subject s08 (held-out).") + "\n" + \
fig("fig_torque", "Joint torque, predicted vs truth (held-out subjects s11, s04, s08).") + "\n" + \
fig("fig_parity", "Parity plots (predicted vs true) over all held-out frames.") + r"""

\section{Sequence models for fatigue and why feature-based wins}\label{sec:ts}
Because fatigue is temporal, we tested sequential deep models (LSTM, PatchTST, TST) with causal sliding windows,
extended features and Optuna tuning. The reference is the final LightGBM (trial \#11), fatigue $R^2=0.936$. None of
the tuned sequential models reaches it:
""" + tbl(tabular(os.path.join(B, "ts_fatigue_tuned.csv"), "Sequential (tuned)"), "Tuned sequential models on fatigue (LOSO).") + r"""
(For reference, default settings: LSTM 0.842, PatchTST 0.814, TST 0.856 --- all below 0.936.)
\paragraph{Theory --- why feature-based wins.}
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt]
  \item \emph{Sufficient statistics.} 3CC makes fatigue an integral of activation; the cumulative features compute
        that integral explicitly, injecting the physics so the model need not learn it.
  \item \emph{Window length.} A $<\!1$\,s window cannot hold the 14--21\,s history; a cumulative feature encodes
        the entire history since $t{=}0$ in one value.
  \item \emph{Data.} Fatigue is slow $\to\sim$1 trajectory per subject ($\sim$8 total) --- far too few for deep
        nets, trivial for a tree $+$ cumulative feature.
\end{enumerate}
A temporal problem does \emph{not} require a sequential model when the form of the temporal dependence is known and
data are limited.
""" + fig("fig_ts_fatigue", "Fatigue: feature-based (LightGBM, trial \\#11) vs sequential deep models.", 0.82) + r"""

\section{Saved model and deployment}
The final model was re-trained on all 8 subjects and serialised:
\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
  \item \texttt{model\_final/lgbm\_trial11.joblib} --- model $+$ input/output scalers $+$ feature/target lists.
  \item \texttt{model\_final/model\_card.json} --- hyper-parameters, feature/target names, LOSO metrics.
  \item \texttt{model\_final/USAGE.py} --- a minimal inference example.
\end{itemize}
Inference: scale the 22 features, call \texttt{model.predict}, inverse-scale $\to$ 13 outputs --- replacing the
OpenSim ID$+$SO$+$3CC chain at run time.

\section{Conclusion and perspectives}
A LightGBM surrogate fed with markerless kinematics, subject anthropometry and physics-based cumulative features
predicts joint torque, muscle forces, activations and fatigue of an unseen subject with a mean LOSO $R^2$ of
$0.95$, effectively replacing the OpenSim ID$+$SO$+$3CC pipeline at inference. Physics-guided feature engineering is
the decisive factor for fatigue, and explainability confirms the model learns physically meaningful relationships.
\paragraph{Perspectives.}
\begin{itemize}[leftmargin=1.4em,itemsep=1pt]
  \item More subjects (hundreds) could make deep sequence models competitive.
  \item Validating activations against EMG (literature references) to certify the labels.
  \item Real-time deployment (the cumulative features are computable online).
\end{itemize}

\end{document}
"""

open(os.path.join(LX, "main.tex"), "w", encoding="utf-8").write(DOC)
print("LaTeX ecrit :", os.path.join(LX, "main.tex"))
print("figures :", len(os.listdir(FG)))
