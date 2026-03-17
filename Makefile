# Variables
SPHINXBUILD   = poetry run sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

# Help target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

# Clean the build directory
clean:
	@echo "Cleaning old builds..."
	rm -rf $(BUILDDIR)

# Build HTML documentation
html:
	@echo "Building HTML documentation with Furo theme..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view."

# Live preview (requires sphinx-autobuild)
live:
	poetry run sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Phony targets (prevents conflict with files named 'clean' or 'html')
.PHONY: help clean html live
