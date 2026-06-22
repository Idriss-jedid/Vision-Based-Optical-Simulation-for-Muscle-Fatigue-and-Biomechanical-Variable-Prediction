# -*- coding: utf-8 -*-
"""Convert Docs/Stage_2_Complete.md to a full, faithful LaTeX document
(Docs/Stage_2_Complete.tex). Copies all text verbatim; converts headings,
pipe-tables, fenced code blocks, lists, bold/italic, and unicode symbols."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "Docs", "Stage_2_Complete.md")
DST = os.path.join(HERE, "..", "Docs", "Stage_2_Complete.tex")

UNICODE = {
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "°": r"\textdegree{}", "²": r"\textsuperscript{2}", "±": r"$\pm$", "×": r"$\times$",
    "≈": r"$\approx$", "≤": r"$\le$", "≥": r"$\ge$", "−": r"$-$", "·": r"$\cdot$",
    "Σ": r"$\Sigma$", "θ": r"$\theta$", "μ": r"$\mu$", "Δ": r"$\Delta$", "∼": r"$\sim$",
    "✅": r"\checkmark{}", "✓": r"\checkmark{}", "⚠️": r"(!)", "⚠": r"(!)",
    "—": "---", "–": "--", "’": "'", "‘": "'", "“": "``", "”": "''", "…": r"\ldots{}",
    "├": "", "└": "", "│": "", "┌": "", "─": "", "░": "", "≃": r"$\simeq$", "§": r"\S{}", "❌": r"$\times$", "⏳": r"(pending)",
    " ": " ", "️": "",
}


def esc(s):
    s = s.replace("\\", "\x01")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("|", r"\textbar{}")]:
        s = s.replace(a, b)
    s = s.replace("~", "\x02").replace("^", "\x03")
    s = s.replace("\x01", r"\textbackslash{}").replace("\x02", r"\textasciitilde{}").replace("\x03", r"\textasciicircum{}")
    for a, b in UNICODE.items():
        s = s.replace(a, b)
    return s


def inline(text):
    store = []

    def stash(kind, val):
        store.append((kind, val))
        return "\x00%d\x00" % (len(store) - 1)

    text = re.sub(r"`([^`]+)`", lambda m: stash("code", m.group(1)), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: stash("b", m.group(1)), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: stash("i", m.group(1)), text)
    text = esc(text)

    def restore(m):
        kind, val = store[int(m.group(1))]
        if kind == "code":
            return r"\texttt{" + esc(val) + "}"
        if kind == "b":
            return r"\textbf{" + esc(val) + "}"
        return r"\textit{" + esc(val) + "}"

    return re.sub("\x00(\\d+)\x00", restore, text)


def col_spec(sep_cells):
    spec = ""
    for c in sep_cells:
        c = c.strip()
        if c.startswith(":") and c.endswith(":"):
            spec += "c"
        elif c.endswith(":"):
            spec += "r"
        else:
            spec += "l"
    return spec


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [c.strip().replace("\\|", "|") for c in cells]


def main():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    out = []
    i = 0
    n = len(lines)
    title_done = False
    list_open = None  # 'itemize' or 'enumerate'

    def close_list():
        nonlocal list_open
        if list_open:
            out.append("\\end{%s}" % list_open)
            list_open = None

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            close_list()
            out.append("\\begin{verbatim}")
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                out.append(lines[i])
                i += 1
            out.append("\\end{verbatim}")
            i += 1
            continue

        # html comment line(s)
        if stripped.startswith("<!--"):
            i += 1
            continue

        # table block
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            close_list()
            header = split_row(lines[i])
            sep = split_row(lines[i + 1])
            base = col_spec(sep)
            rows = []
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            ncol = len(header)
            # per-column max raw length (header + body) decides wrapping
            THRESH = 16
            cols = []
            for c in range(ncol):
                ml = len(header[c]) if c < len(header) else 0
                for r in rows:
                    if c < len(r):
                        ml = max(ml, len(r[c]))
                align = base[c] if c < len(base) else "l"
                if ml > THRESH:
                    if align == "r":
                        cols.append(r">{\raggedleft\arraybackslash}X")
                    else:
                        cols.append(r">{\raggedright\arraybackslash}X")
                else:
                    cols.append(align)
            use_x = any("X" in c for c in cols)
            spec = "".join(cols)
            resize = (not use_x) and ncol >= 6   # guard wide all-short tables
            out.append("\\begin{table}[h]\\centering\\footnotesize")
            if resize:
                out.append("\\resizebox{\\textwidth}{!}{%")
            if use_x:
                out.append("\\begin{tabularx}{\\textwidth}{%s}" % spec)
            else:
                out.append("\\begin{tabular}{%s}" % spec)
            out.append("\\toprule")
            out.append(" & ".join(inline(c) for c in header) + " \\\\")
            out.append("\\midrule")
            for r in rows:
                while len(r) < ncol:
                    r.append("")
                out.append(" & ".join(inline(c) for c in r[:ncol]) + " \\\\")
            out.append("\\bottomrule")
            out.append("\\end{tabularx}" if use_x else "\\end{tabular}")
            if resize:
                out.append("}")
            out.append("\\end{table}")
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list()
            level = len(m.group(1))
            htext = m.group(2).strip().rstrip("#").strip()
            if "=== PART" in htext or htext.upper().startswith("PART"):
                part = re.sub(r"=+\s*", "", htext).strip()
                out.append("\\clearpage")
                out.append("\\section{%s}" % inline(part))
            elif level == 1 and not title_done:
                title_done = True  # first H1 -> document title (set in preamble)
            elif level == 1:
                out.append("\\subsection{%s}" % inline(htext))
            elif level == 2:
                out.append("\\subsubsection{%s}" % inline(htext))
            elif level == 3:
                out.append("\\paragraph{%s}~\\\\" % inline(htext))
            else:
                out.append("\\subparagraph{%s}~\\\\" % inline(htext))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            close_list()
            out.append("\\medskip\\hrule\\medskip")
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            close_list()
            out.append("\\begin{quote}")
            while i < n and lines[i].strip().startswith(">"):
                out.append(inline(lines[i].strip()[1:].strip()))
                i += 1
            out.append("\\end{quote}")
            continue

        # lists
        mu = re.match(r"^\s*[-*]\s+(.*)$", line)
        mo = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if mu or mo:
            want = "itemize" if mu else "enumerate"
            if list_open != want:
                close_list()
                out.append("\\begin{%s}" % want)
                list_open = want
            out.append("\\item " + inline((mu or mo).group(1)))
            i += 1
            continue

        # list continuation: indented wrapped text belonging to the current item
        if list_open and stripped and line[:1] in (" ", "\t"):
            out[-1] = out[-1] + " " + inline(stripped)
            i += 1
            continue

        # blank line
        if stripped == "":
            close_list()
            out.append("")
            i += 1
            continue

        # normal paragraph text
        close_list()
        out.append(inline(line))
        i += 1

    close_list()
    body = "\n".join(out)

    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\title{Stage 2 --- Biomechanical Analysis: Complete}
\author{Vision-Based Optical Simulation}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""
    text = preamble + body + "\n\\end{document}\n"
    open(DST, "w", encoding="utf-8", newline="\n").write(text)
    print("Wrote %s : %d KB, %d lines" % (DST, os.path.getsize(DST) // 1024, text.count("\n") + 1))


if __name__ == "__main__":
    main()
