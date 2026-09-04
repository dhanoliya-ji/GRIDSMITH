# gridsmith -- deterministic PDF table extraction.
#
# Two build modes, because §3.1 asks for an image that builds and runs with no network
# access, and honesty about that is better than a claim that quietly needs a mirror:
#
#   docker build -t gridsmith .                     # default: fetches apt + pip
#   docker build -t gridsmith --network none \
#          --build-arg OFFLINE=1 .                 # needs vendor/ prepopulated first
#
# The *run* is offline in both modes, and `--offline` installs a socket guard that
# raises on any connection attempt, so "it never touched the network" is enforced
# rather than asserted.  See README "Offline and determinism".
#
# To prepare an offline build:  bash tools/vendor_deps.sh
FROM python:3.11-slim-bookworm

ARG OFFLINE=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=20240928 \
    PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app
COPY requirements.txt ./
COPY vendor/ /vendor/

# tesseract is optional: without it the pipeline still runs and flags raster cells
# OCR_UNAVAILABLE instead of guessing.  Installed here so the OCR path is exercised.
RUN if [ "$OFFLINE" = "1" ]; then \
        (ls /vendor/*.deb >/dev/null 2>&1 && dpkg -i /vendor/*.deb || \
         echo "no vendored debs: continuing without tesseract") && \
        pip install --no-index --find-links=/vendor -r requirements.txt ; \
    else \
        apt-get update && \
        apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev && \
        rm -rf /var/lib/apt/lists/* && \
        pip install --no-cache-dir -r requirements.txt ; \
    fi

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY metrics/ ./metrics/
COPY tests/ ./tests/
COPY tools/ ./tools/
RUN pip install --no-deps --no-index -e . 2>/dev/null || \
    ln -s /app/src/gridsmith /usr/local/lib/python3.11/site-packages/gridsmith

ENV PYTHONPATH=/app/src
RUN python -c "import gridsmith, pymupdf, numpy, openpyxl; print('gridsmith', gridsmith.__version__, 'ok')"

# Sanity check that needs no input document: the fixtures are generated, extracted and
# scored entirely inside the image, so `docker run` proves the pipeline works offline.
RUN python metrics/fixtures/make_fixtures.py >/dev/null && \
    python -m gridsmith run --input metrics/fixtures/f5_watermark_typo.pdf \
        --out /tmp/_smoke --offline --quiet && \
    python -c "import json;m=json.load(open('/tmp/_smoke/metrics.json'));assert m['counts']['logical_tables']==1;print('smoke ok')" && \
    rm -rf /tmp/_smoke

ENTRYPOINT ["python", "-m", "gridsmith"]
CMD ["--help"]
