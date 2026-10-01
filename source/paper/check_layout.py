#!/usr/bin/env python3
"""Check the built PDF's geometry, not just LaTeX's opinion of it.

    ./.venv312/bin/python paper/check_layout.py

LaTeX reported zero Overfull boxes while the author block ran off the right
edge of page 1: the block is a centred box that does not wrap, so nothing
overflowed a paragraph and nothing was reported. A first attempt at catching it
here silently matched nothing and printed an empty table, which read as "clean".
This measures the ink instead: every text line's bounding box against the text
block, taken from the PDF.
"""
import re
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "main.pdf")
PT = 72.0
fails = []


def boxes(page):
    """(xmin, ymin, xmax, ymax, text) for every line on a page."""
    xml = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page),
                          "-bbox-layout", PDF, "-"],
                         capture_output=True, text=True).stdout
    out = []
    for m in re.finditer(
            r'<line xMin="([\d.-]+)" yMin="([\d.-]+)" '
            r'xMax="([\d.-]+)" yMax="([\d.-]+)">(.*?)</line>', xml, re.S):
        x0, y0, x1, y1, body = m.groups()
        txt = re.sub(r"<[^>]+>", "", body).strip()
        out.append((float(x0), float(y0), float(x1), float(y1), txt))
    return out


npages = int(re.search(r"Pages:\s+(\d+)",
                       subprocess.run(["pdfinfo", PDF], capture_output=True,
                                      text=True).stdout).group(1))
size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)",
                 subprocess.run(["pdfinfo", PDF], capture_output=True,
                                text=True).stdout)
W, H = float(size.group(1)), float(size.group(2))

# The text block, taken from a body page rather than assumed.
body = boxes(3)
LEFT = min(b[0] for b in body)
RIGHT = max(b[2] for b in body)
print(f"  page {W:.0f}x{H:.0f} pt ({W/PT:.1f}x{H/PT:.1f} in), {npages} pages")

# The page count was printed and never asserted, so a nine-page build passed a
# gate whose whole job is the venue's page limit -- and did, for three
# manuscript commits. CBDCom regular includes eight pages; a ninth is $100 and
# ten is the hard ceiling, so overflow is a decision to be taken deliberately,
# not something a gate should let through silently.
PAGE_BUDGET = 8
if npages > PAGE_BUDGET:
    print(f"  [FAIL] {npages} pages exceeds the included budget of "
          f"{PAGE_BUDGET}")
    fails.append("page budget")
else:
    print(f"  [PASS] within the {PAGE_BUDGET}-page budget")

# And the PDF has to be the one this .tex produces. The committed build had
# been three manuscript commits behind, so the layout gate was measuring a
# document the number gate had never read.
_tex_mtime = os.path.getmtime(os.path.join(HERE, "main.tex"))
if os.path.getmtime(PDF) < _tex_mtime:
    print("  [FAIL] main.pdf is older than main.tex; the gates are looking at "
          "two different documents")
    fails.append("stale pdf")
else:
    print("  [PASS] the PDF was built from the current source")

# Undefined references and overfull boxes are reported by LaTeX and were being
# read by a human, which is how a dangling \ref reached the PDF as "??" and a
# ninth page reached three commits.
_log = os.path.join(HERE, "main.log")
if os.path.exists(_log):
    _txt = open(_log, errors="replace").read()
    _undef = len(re.findall(r"Reference `[^']+' on page \d+ undefined", _txt))
    _over = len(re.findall(r"^Overfull \\hbox", _txt, re.M))
    for _label, _n in (("undefined references", _undef), ("overfull boxes", _over)):
        print(f"  [{'PASS' if _n == 0 else 'FAIL'}] no {_label}" +
              (f"  ({_n} in main.log)" if _n else ""))
        if _n:
            fails.append(_label)
else:
    print("  [FAIL] main.log is missing; build with --keep-logs")
    fails.append("no build log")

# The rendered text must not contain LaTeX's unresolved-reference marker.
_all_text = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                           text=True).stdout
if "??" in _all_text:
    print("  [FAIL] the rendered PDF contains '??', an unresolved reference")
    fails.append("rendered ??")
else:
    print("  [PASS] no unresolved reference marker in the rendered text")
print(f"  text block {LEFT:.1f}--{RIGHT:.1f} pt "
      f"(margins {LEFT/PT:.2f}/{(W-RIGHT)/PT:.2f} in)")

TOL = 1.0
bad = []
for pg in range(1, npages + 1):
    for x0, y0, x1, y1, txt in boxes(pg):
        if x1 > RIGHT + TOL or x0 < LEFT - TOL:
            bad.append((pg, x0, x1, txt[:60]))
if bad:
    fails.append("overflow")
    print(f"  [FAIL] {len(bad)} line(s) outside the text block:")
    for pg, x0, x1, txt in bad[:8]:
        print(f"          p{pg} [{x0:.0f},{x1:.0f}] {txt!r}")
else:
    print(f"  [PASS] every line on all {npages} pages sits inside the text block")

# US Letter, two columns, and the column geometry the IEEE conference format
# produces. These come from IEEEtran's defaults; the point of checking is that
# nothing in the document has overridden them.
letter = abs(W - 612) < 1 and abs(H - 792) < 1
print(f"  [{'PASS' if letter else 'FAIL'}] US Letter")
if not letter:
    fails.append("paper size")

mid = (LEFT + RIGHT) / 2
lcol = [b for b in body if b[0] < mid]
rcol = [b for b in body if b[0] >= mid]
colw = max(b[2] for b in lcol) - min(b[0] for b in lcol)
gutter = min(b[0] for b in rcol) - max(b[2] for b in lcol)
ok_col = abs(colw / PT - 3.5) < 0.1
print(f"  [{'PASS' if ok_col else 'FAIL'}] column width {colw/PT:.2f} in "
      f"(IEEE conference: 3.5 in), gutter {gutter/PT:.2f} in, two columns")
if not ok_col:
    fails.append("column width")

# The IEEE conference template sets the abstract in bold.
p1 = subprocess.run(["pdftotext", "-f", "1", "-l", "1", PDF, "-"],
                    capture_output=True, text=True).stdout
fonts = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
# Nimbus calls its bold weight "Medi"; grepping for "Bold" alone reported no
# bold face in a PDF that had one, which is the failure mode this file is for.
has_bold = bool(re.search(r"Bold|Medi|-Bd\b|bx\b", fonts))
print(f"  [{'PASS' if has_bold else 'FAIL'}] a bold face is embedded")
if not has_bold:
    fails.append("no bold face")

# IEEEtran asks for Times; if the engine cannot find it, it substitutes
# silently and every \textbf in the paper stops being bold.
serif = re.search(r"(NimbusRom|Times|Termes|NimbusSerif|ntx|ptm)", fonts)
subbed = re.search(r"(LMRoman|CMR\d|LatinModern)", fonts)
print(f"  [{'PASS' if serif else 'FAIL'}] a Times face is embedded "
      f"({serif.group(1) if serif else 'none'})")
if not serif:
    fails.append("not Times")
undef = subprocess.run(["grep", "-coE", r"Font shape .[A-Z0-9]+/ptm/[a-z]+/[a-z]+. undefined",
                        os.path.join(HERE, "main.log")],
                       capture_output=True, text=True).stdout.strip()
body_lm = bool(re.search(r"LMRoman", fonts))
print(f"  [{'PASS' if not body_lm else 'FAIL'}] no Latin Modern fallback in the "
      f"body ({undef} undefined ptm shape lines in the log)")
if body_lm:
    fails.append("latin modern fallback")
print(f"  [{'PASS' if 'Abstract—' in p1 or 'Abstract-' in p1 else 'FAIL'}] "
      f"abstract present and labelled")

# IEEE's PDF requirements ask for no link annotations and no bookmark tree.
# hyperref emits both by default and \hypersetup{hidelinks} removes only the
# coloured frame, so the file kept 78 /Link objects and an outline that nothing
# on the page revealed. A raw byte search for "/Link" finds nothing either:
# PDF 1.5 stores those objects inside compressed object streams. Every stream
# has to be inflated before the question can even be asked.
import zlib

raw = open(PDF, "rb").read()
counts = {"/Outlines": 0, "/Annots": 0, "/Link": 0}
for m in re.finditer(rb"stream\r?\n", raw):
    start = m.end()
    end = raw.find(b"endstream", start)
    if end < 0:
        continue
    try:
        data = zlib.decompress(raw[start:end])
    except zlib.error:
        continue                      # not a deflate stream (fonts, images)
    for key in counts:
        counts[key] += data.count(key.encode())
for key in counts:
    counts[key] += raw.count(key.encode())
clean = not any(counts.values())
print(f"  [{'PASS' if clean else 'FAIL'}] no links or bookmarks "
      f"(link={counts['/Link']} outline={counts['/Outlines']} "
      f"annot={counts['/Annots']})")
if not clean:
    fails.append("pdf has links/bookmarks")

# Fonts must be embedded and no Type 3 may appear; both are Xplore conditions
# and both are properties of the built file, not of the source.
noemb = [ln for ln in fonts.splitlines()[2:] if ln.strip() and " no " in ln]
type3 = [ln for ln in fonts.splitlines()[2:] if "Type 3" in ln]
ok_fonts = not noemb and not type3
print(f"  [{'PASS' if ok_fonts else 'FAIL'}] every font embedded, no Type 3 "
      f"({len([l for l in fonts.splitlines()[2:] if l.strip()])} faces)")
if not ok_fonts:
    fails.append("font embedding")

print(f"\n==== check_layout: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)} ====")
sys.exit(0 if not fails else 1)
