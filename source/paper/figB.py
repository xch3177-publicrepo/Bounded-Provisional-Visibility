"""Emit Figure B (end-to-end poison exposure) as a TikZ fragment.

    ./.venv312/bin/python paper/figB.py [results/W2-milvus.json]

Unlike figA.py, which carries its series as literals copied off a printed
table, this reads `prototype/results/W2-*.json` directly: the figure cannot
drift out of step with the numbers because there is only one copy of them.
Re-run after any re-run of poison_exposure.py and paste figB.tex over the
tikzpicture in main.tex.

PANELS selects which backlog conditions get a panel. Only "heavy" is drawn: the
normal-backlog panel is three lines on zero and two identical short pulses, so
its content is "nothing happens" -- a sentence, not a picture -- and the page
budget is hard.

Drawn in TikZ rather than imported as an image so the build stays reproducible
from source (no matplotlib in the venv pinned for pymilvus 2.4, no binary asset
that can go stale).
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "prototype", "results", "W2-milvus.json")

W, PH, GAP = 62.0, 24.0, 9.0        # mm: width, panel height, gap between panels
# One panel, not two. The normal-backlog panel is three lines on zero and two
# identical short pulses -- its content is "nothing happens", which is a
# sentence, not a picture, and the page budget is hard. The heavy-backlog
# panel is where the baselines separate.
PANELS = ["heavy"]
TMAX = None                          # filled from the config
QSET = "craft"                        # the crafted set: the exposure WINDOW is a
                                     # protocol property and shows most clearly
                                     # where the attack actually lands; transfer
                                     # to non-crafted queries is a table number
STYLE = {"B1": ("black", "solid"), "B2": ("black!45", "densely dotted"),
         "B3": ("black!45", "dashed"), "B4": ("black", "solid")}
WIDTH = {"B1": ".7pt", "B2": ".5pt", "B3": ".5pt", "B4": "1.1pt"}


def load(path):
    with open(path) as f:
        return json.load(f)


def curve(doc, baseline, backlog, tp):
    """PRR per time bin, pooled across seeds, for one baseline/backlog cell.

    Re-binned to one full query rotation. The run emits 0.25 s bins, but the
    craft set cycles eight distinct queries at eight per second, so a 0.25 s
    window holds only two of them: whether that pair happens to include the one
    query the attack never reached decides between a bin reading 1.0 and one
    reading 0.5, and the curve becomes a sawtooth that is an artefact of the
    rotation rather than anything about exposure. A bin one rotation wide holds
    each query exactly once, which removes the artefact instead of smoothing
    over it. It is also the honest time resolution: below one rotation the rate
    is undersampled by construction.

    Pooled by summing hits and queries rather than taking a median of rates --
    a median of per-seed rates would reintroduce the same quantisation.
    """
    cfg = doc["config"]
    cells = [c for c in doc["metrics"]["cells"]
             if c["baseline"] == baseline and c["backlog"] == backlog
             and c["Tp"] == tp]
    if not cells:
        return []
    rotation = cfg[f"n_{'craft' if QSET == 'craft' else 'target'}_q"] / (cfg["qps"] / 3.0)
    per_bin = {}
    for c in cells:
        for pt in c["sets"][QSET]["series"]:
            b = int(pt["t"] / rotation)
            hits, n = per_bin.get(b, (0.0, 0))
            per_bin[b] = (hits + pt["prr"] * pt["n"], n + pt["n"])
    return [(b * rotation, hits / n) for b, (hits, n) in sorted(per_bin.items()) if n]


def emit(doc):
    cfg = doc["config"]
    tp, inject, dur = cfg["tp"], cfg["inject_at"], cfg["dur"]
    global TMAX
    last = max(t for name in PANELS for b in STYLE
               for t, _ in curve(doc, b, name, tp) or [(inject, 0)])
    TMAX = last - inject

    def X(t):                        # time is plotted from the injection instant
        return max(0.0, (t - inject)) / TMAX * W

    def Y(p, base):
        return base + p * PH

    L = [r"\begin{tikzpicture}[font=\scriptsize,inner sep=0pt]"]
    titles = {"heavy": "heavy verifier backlog", "normal": "verifier keeps up"}
    panels = [(name, i * (PH + GAP), titles[name])
              for i, name in enumerate(reversed(PANELS))]
    for backlog, base, title in panels:
        L.append(r"  \draw[black!25] (0,%.2fmm) rectangle (%.2fmm,%.2fmm);"
                 % (base, W, base + PH))
        for p in (0.5,):
            L.append(r"  \draw[black!12] (0,%.2fmm) -- (%.2fmm,%.2fmm);"
                     % (Y(p, base), W, Y(p, base)))
        for lab, p in (("0", 0.0), (".5", 0.5), ("1", 1.0)):
            L.append(r"  \node[left=1.5pt] at (0,%.2fmm) {%s};" % (Y(p, base), lab))
        # The deadline, marked once per panel: it is where B4 is supposed to cut.
        L.append(r"  \draw[black!35,densely dashed,line width=.4pt] "
                 r"(%.2fmm,%.2fmm) -- (%.2fmm,%.2fmm);"
                 % (X(inject + tp), base, X(inject + tp), base + PH))
        # The T_p label has nowhere good above or below: the curves run at 0.85
        # and 0 where the deadline line crosses them, and putting it outside the
        # frame lands it in the legend. Mid-height, just right of the line, is
        # the one clear pocket -- B4 has already dropped and B1/B3 are far above.
        L.append(r"  \node[anchor=west,black!55,font=\tiny] at "
                 r"(%.2fmm,%.2fmm) {$T_p$};" % (X(inject + tp) + 0.7, Y(0.5, base)))
        # No in-panel title: with a single panel it only repeats the caption, and
        # it sat on top of the B1 curve.
        for b in ("B3", "B2", "B4", "B1"):
            pts = curve(doc, b, backlog, tp)
            if not pts:
                continue
            pts = [(t, p) for t, p in pts if t >= inject - 1e-9]
            path = " -- ".join("(%.2fmm,%.2fmm)" % (X(t), Y(p, base))
                               for t, p in pts)
            col, dash = STYLE[b]
            L.append(r"  \draw[%s,%s,line width=%s] %s;"
                     % (col, dash, WIDTH[b], path))
    # x axis on the lower panel only
    for t in range(0, int(TMAX) + 1):
        L.append(r"  \node[below=1pt,font=\tiny] at (%.2fmm,0) {%d};"
                 % (t / TMAX * W, t))
    L.append(r"  \node[below=5pt] at (%.2fmm,0) "
             r"{time since ingestion (s)};" % (W / 2))
    L.append(r"  \node[rotate=90,anchor=south] at (-5.2mm,%.2fmm) "
             r"{poisoned top-$k$ rate};" % ((len(PANELS) * PH + (len(PANELS) - 1) * GAP) / 2))
    # Legend above the panel, two rows. A single row on a fixed pitch put
    # "B3 no deadline" under the next entry's sample line: at \tiny that label
    # is about 19 mm wide and the pitch was 15 mm. Rows are laid out by advancing
    # past each entry's estimated width rather than by a constant step.
    top = len(PANELS) * PH + (len(PANELS) - 1) * GAP
    entries = [("B1", "B1 none"), ("B2", "B2 pre-vet"),
               ("B3", "B3 no deadline"), ("B4", "B4 ours")]
    SAMPLE, PAD, CHAR = 4.0, 0.6, 1.05      # mm; CHAR is \tiny advance width
    for row in (0, 1):
        x = 0.0
        yleg = top + 6.4 - row * 3.2
        for b, lab in entries[2 * row:2 * row + 2]:
            col, dash = STYLE[b]
            L.append(r"  \draw[%s,%s,line width=%s] (%.2fmm,%.2fmm) -- (%.2fmm,%.2fmm);"
                     % (col, dash, WIDTH[b], x, yleg, x + SAMPLE, yleg))
            L.append(r"  \node[anchor=west,font=\tiny] at (%.2fmm,%.2fmm) {%s};"
                     % (x + SAMPLE + PAD, yleg, lab))
            x += SAMPLE + PAD + CHAR * len(lab) + 4.0
    L.append(r"\end{tikzpicture}")
    return L


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    doc = load(path)
    lines = emit(doc)
    out = os.path.join(HERE, "figB.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{path}\n  evidence_level={doc['evidence_level']} "
          f"admissible={doc['admissible']}")
    print(f"  wrote {out}  ({len(lines)} lines)")
