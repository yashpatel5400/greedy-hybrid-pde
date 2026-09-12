#!/bin/zsh
# Build in a non-synced scratch directory (the repo lives under iCloud-synced
# ~/Documents, where auxiliary files were intermittently corrupted), then copy
# the PDF back next to the sources.
SRC="$(cd "$(dirname "$0")" && pwd)"
BUILD="${PAPER_BUILD_DIR:-/private/tmp/claude-501/paperbuild}"
mkdir -p "$BUILD"
rsync -a --delete --exclude '*.aux' --exclude '*.log' --exclude '*.out' --exclude '*.pdf' --exclude '*.fls' --exclude '*.fdb_latexmk' --exclude '*.blg' --exclude '*.bbl' --exclude 'ICLR 2026 (old)' --exclude 'Old Writeups' "$SRC/" "$BUILD/"
cd "$BUILD"
pdflatex -interaction=nonstopmode neurips_2026.tex > /dev/null 2>&1
bibtex neurips_2026 > /dev/null 2>&1
pdflatex -interaction=nonstopmode neurips_2026.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode neurips_2026.tex > /dev/null 2>&1
echo "errors: $(grep -c '^!' neurips_2026.log)  undefined-refs: $(grep -c 'undefined references' neurips_2026.log)  $(grep -o 'Output written on neurips_2026.pdf ([0-9]* pages' neurips_2026.log | grep -o '[0-9]* pages')"
cp neurips_2026.pdf "$SRC/neurips_2026.pdf" && cp neurips_2026.aux "$SRC/neurips_2026.aux" && echo "PDF copied to $SRC/neurips_2026.pdf"
