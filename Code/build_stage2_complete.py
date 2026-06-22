"""Concatenate the full content of the Stage-2 analysis/validation notes into one
detailed document, Docs/Stage_2_Complete.md (verbatim copy of each)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "Docs")

PARTS = [
    ("Stage_2_Biomechanical_Analysis.md", "Biomechanical Analysis - master (methods, multi-condition results, validation)"),
    ("Stage_2_ID_SO_Results.md",          "Inverse Dynamics + Static Optimization"),
    ("Stage_2_Fatigue_3CC.md",            "3CC-Coupled Static Optimization - the Fatigue Label"),
    ("Stage_2_Experimental_Validation.md","Experimental Validation - All Muscles + Fatigue Endurance"),
    ("Stage_2_CMC_CrossCheck.md",         "CMC Cross-Check of the SO Labels"),
    ("Stage_2_EMG_MVC_Validation.md",     "EMG / MVC Validation (honest split)"),
]

out = []
out.append("# Stage 2 - Biomechanical Analysis: COMPLETE (consolidated, full detail)\n")
out.append("This single document contains the **full content** of every Stage-2 analysis &")
out.append("validation note, copied verbatim and combined. It answers the question \"focus on")
out.append("Biomechanical Analysis - which to use (ID / SO / CMC) and why, for a fatigue study\"")
out.append("and carries through to the complete validation suite.\n")
out.append("## Table of contents\n")
for i, (fn, title) in enumerate(PARTS, 1):
    out.append("%d. **%s**  (source: %s)" % (i, title, fn))
out.append("\n> Each part below is the complete original note, unabridged.\n")

for i, (fn, title) in enumerate(PARTS, 1):
    body = open(os.path.join(DOCS, fn), encoding="utf-8").read().rstrip()
    out.append("\n\n<!-- ================================================================ -->")
    out.append("# === PART %d - %s ===" % (i, title))
    out.append("<!-- source: %s -->\n" % fn)
    out.append(body)

text = "\n".join(out) + "\n"
dst = os.path.join(DOCS, "Stage_2_Complete.md")
open(dst, "w", encoding="utf-8", newline="\n").write(text)
print("Wrote %s : %d KB, %d lines" % (dst, os.path.getsize(dst) // 1024, text.count("\n") + 1))
for fn, _ in PARTS:
    n = open(os.path.join(DOCS, fn), encoding="utf-8").read().count("\n") + 1
    print("  + %-40s %5d lines" % (fn, n))
