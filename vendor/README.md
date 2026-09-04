# vendor/

Populate this directory to build the image with `--network none`:

    bash tools/vendor_deps.sh

It downloads the pinned wheels from `requirements.txt` plus the `tesseract-ocr` .deb and
its dependencies. Left empty, the default (networked) build path is used instead and the
image still runs fully offline -- only the *build* differs.

Kept in git as an empty directory with this note so `COPY vendor/` in the Dockerfile
never fails.
