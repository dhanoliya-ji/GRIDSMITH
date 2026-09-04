import pymupdf, sys, collections, os, hashlib
for path in sys.argv[1:]:
    d = pymupdf.open(path)
    print("="*80)
    print(os.path.basename(path), "pages=", d.page_count, "sha256=", hashlib.sha256(open(path,'rb').read()).hexdigest()[:12])
    print("meta:", {k:v for k,v in d.metadata.items() if v})
    for i,p in enumerate(d):
        t = p.get_text()
        nchar = len(t.strip())
        imgs = p.get_images(full=True)
        try: drw = p.get_drawings()
        except Exception: drw = []
        nline = sum(1 for dd in drw for it in dd["items"] if it[0]=="l")
        nre   = sum(1 for dd in drw for it in dd["items"] if it[0]=="re")
        blocks = p.get_text("blocks")
        print(f"  p{i+1:>3} rot={p.rotation:>3} mbox={tuple(round(x) for x in p.rect)} chars={nchar:>5} imgs={len(imgs):>2} draw={len(drw):>4} L={nline:>4} R={nre:>4} blocks={len(blocks):>3}")
