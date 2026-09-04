#!/usr/bin/env bash
# Populate vendor/ so `docker build --network none --build-arg OFFLINE=1 .` works.
#
# Run once, on a machine with network access, before an air-gapped build.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor

echo "== python wheels =="
python -m pip download -r requirements.txt -d vendor

echo "== tesseract debs =="
if command -v apt-get >/dev/null 2>&1; then
  ( cd vendor && apt-get download tesseract-ocr tesseract-ocr-eng libtesseract5 \
      libleptonica-dev liblept5 2>/dev/null || \
    echo "apt-get download failed; the image will build without tesseract" )
else
  echo "not a Debian host: fetch the tesseract .deb files manually into vendor/"
fi

echo
echo "vendor/ now holds $(ls vendor | wc -l) files"
echo "build with: docker build -t gridsmith --network none --build-arg OFFLINE=1 ."
