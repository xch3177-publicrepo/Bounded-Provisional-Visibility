"""Emit the freshness-exposure characterisation as a TikZ fragment.

    NOT IN THE MANUSCRIPT any more. This figure was Fig. 1 of the results until
    2026-07-27, when the eight-page budget made it compete with the
    poisoning-exposure curve (paper/figB.py) and lost: the trade-off it drew
    from the E1 simulation is now shown empirically, on both backends, by the
    W2 experiment. The sweep behind it is still a real E1 result -- see C10 in
    the claim-evidence matrix -- so the generator stays rather than being
    deleted. Nothing in main.tex references it; check before reinstating that
    the page budget can carry it.

    ./.venv312/bin/python paper/figA.py && cat paper/figA.tex

The data points below are read off `prototype/pareto.py`, which prints the E1
harness sweep as a table and draws nothing. Re-run it and update the series here
if the sweep changes, then paste the fragment over the tikzpicture in main.tex.

Drawn in TikZ rather than imported as an image so the build stays reproducible
from source: no matplotlib (which would pull numpy into the venv pinned for
pymilvus 2.4) and no binary asset that can drift out of sync with the numbers.
Kept in the repository rather than a scratch directory -- a figure whose
generator only exists in a session is a figure that cannot be regenerated.
"""
import math
import os
W,H = 62.0, 30.0                     # mm plot area
YMAX = 1.12                          # headroom so an A_f=1.0 label clears the frame
XLO,XHI = math.log10(0.085), math.log10(5.6)
def X(e): return (math.log10(e)-XLO)/(XHI-XLO)*W
def Y(a): return a/YMAX*H
series = [
 ("fast",       "solid",      "*",        [(0.100,0.983),(0.200,1.000)]),
 ("moderate",   "dashed",     "square*",  [(0.100,0.880),(0.250,0.905),(0.500,0.938),(1.000,1.000)]),
 ("overloaded", "solid",      "triangle*",[(0.100,0.078),(0.250,0.103),(0.500,0.131),(1.000,0.193),(1.985,0.316),(4.820,0.669)]),
 ("burst",      "densely dotted","diamond*",[(0.100,0.047),(0.250,0.072),(0.500,0.097),(1.000,0.171),(1.990,0.281),(4.900,0.652)]),
]
col = {"fast":"black!55","moderate":"black","overloaded":"black","burst":"black!55"}
L=[]
L.append(r"\begin{tikzpicture}[font=\scriptsize,inner sep=0pt]")
L.append(r"  \draw[black!25] (0,0) rectangle (%.2fmm,%.2fmm);" % (W,H))
for a in (0.5,):
    L.append(r"  \draw[black!12] (0,%.2fmm) -- (%.2fmm,%.2fmm);" % (Y(a),W,Y(a)))
for e in (0.1,0.2,0.5,1,2,5):
    L.append(r"  \draw[black!12] (%.2fmm,0) -- (%.2fmm,%.2fmm);" % (X(e),X(e),H))
    L.append(r"  \node[below=1pt] at (%.2fmm,0) {%s};" % (X(e), ("%g"%e)))
for a,lab in ((0,"0"),(0.5,".5"),(1.0,"1")):
    L.append(r"  \node[left=1.5pt] at (0,%.2fmm) {%s};" % (Y(a),lab))
for name,style,mark,pts in series:
    p = " -- ".join("(%.2fmm,%.2fmm)" % (X(e),Y(a)) for e,a in pts)
    L.append(r"  \draw[%s,%s,line width=.5pt] %s;" % (col[name],style,p))
    for e,a in pts:
        L.append(r"  \fill[%s] (%.2fmm,%.2fmm) circle (.5pt);" % (col[name],X(e),Y(a)))
L.append(r"  \node[anchor=south,%s] at (%.2fmm,%.2fmm) {fast};" % (col['fast'],X(0.145),Y(1.00)))
L.append(r"  \node[anchor=north west,%s] at (%.2fmm,%.2fmm) {moderate};" % (col['moderate'],X(0.30),Y(0.90)))
L.append(r"  \node[anchor=south east,%s] at (%.2fmm,%.2fmm) {overloaded};" % (col['overloaded'],X(4.10),Y(0.60)))
L.append(r"  \node[anchor=north east,%s] at (%.2fmm,%.2fmm) {burst};" % (col['burst'],X(3.60),Y(0.40)))
L.append(r"  \node[below=7pt] at (%.2fmm,0) {unvetted exposure budget $\Eu$ (s, log scale)};" % (W/2))
L.append(r"  \node[rotate=90,anchor=south] at (-5.0mm,%.2fmm) {clean avail.\ $A_f$};" % (H/2))
L.append(r"\end{tikzpicture}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figA.tex"), "w").write("\n".join(L)+"\n")
print("\n".join(L[:4])); print("... %d lines" % len(L))
