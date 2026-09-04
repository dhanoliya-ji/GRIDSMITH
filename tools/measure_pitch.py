import sys
sys.path.insert(0, "src")
from gridsmith import s1_intake as s1
from gridsmith.geometry import median
from gridsmith.textlayer import extract_page_text

SD = "c:/Users/gajen/OneDrive/Desktop/sample data/sample data/"
cases = [("Apple Form 10-K.pdf", 25, 118.0, 172.0, "Apple: 5 SEPARATE rows"),
         ("Shell MRPL Aviation Fuels and Services Limited.pdf", 4, 250.0, 294.0,
          "ICRA: 1 row WRAPPED over 5 lines"),
         ("Shell MRPL Aviation Fuels and Services Limited.pdf", 1, 213.6, 236.1,
          "ICRA: 1 row wrapped (summary)"),
         ("Apple Form 10-K.pdf", 32, 149.0, 200.0, "Apple: separate rows p32")]
for name, pno, y0, y1, label in cases:
    d = s1.open_document(SD + name)
    pt = extract_page_text(d[pno - 1], pno)
    band = [ln for ln in pt.lines if y0 <= (ln.y0 + ln.y1) / 2 <= y1]
    if len(band) < 2:
        print("%-40s too few lines" % label)
        continue
    band.sort(key=lambda l: l.baseline)
    gaps = [b.baseline - a.baseline for a, b in zip(band, band[1:])]
    h = median([ln.y1 - ln.y0 for ln in band])
    sz = median([ln.size for ln in band])
    print("%-38s n=%d  pitch=%.2f  glyph_h=%.2f  font=%.1f  pitch/h=%.3f  pitch/font=%.3f"
          % (label, len(band), median(gaps), h, sz, median(gaps) / h, median(gaps) / sz))
    print("      first lines: %s" % [ln.text[:26] for ln in band[:3]])
