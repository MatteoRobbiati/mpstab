# Minimal Sphinx Makefile for the mpstab docs.
#
# Usage (from the repo root):
#   make -C docs html        # incremental HTML build
#   make -C docs rebuild     # clean + full HTML build
#   make -C docs open        # build and open in the browser (macOS)
#   make -C docs live        # live-reload server (needs sphinx-autobuild)
#   make -C docs strict      # fail on warnings (-W --keep-going)
#   make -C docs doctest     # run doctests embedded in the docs
#   make -C docs clean       # remove the build directory
#
# Requires: `uv sync --group docs` and a system pandoc (`brew install pandoc`).

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html rebuild open live strict doctest clean check-pandoc Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo ""
	@echo "Extra targets: rebuild, open, live, strict"

check-pandoc:
	@command -v pandoc >/dev/null 2>&1 || { \
		echo "pandoc not found: install it with 'brew install pandoc' (needed by nbsphinx)"; \
		exit 1; }

html: check-pandoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

rebuild: clean
	@$(MAKE) html SPHINXOPTS="$(SPHINXOPTS) -E"

open: html
	@open "$(BUILDDIR)/html/index.html"

live: check-pandoc
	@uv run --with sphinx-autobuild sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--watch ../src --ignore "$(BUILDDIR)/*" $(SPHINXOPTS)

strict: check-pandoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS)

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

# Catch-all: forward any other target (latexpdf, linkcheck, ...) to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
