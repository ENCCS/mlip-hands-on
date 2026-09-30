SPHINXBUILD ?= sphinx-build
AUTOBUILD ?= sphinx-autobuild
SOURCEDIR = content
BUILDDIR = _build
PORT ?= 8766

.PHONY: html livehtml
html:
	$(SPHINXBUILD) -n -W -b html $(SOURCEDIR) $(BUILDDIR)/html

livehtml:
	$(AUTOBUILD) -n -W --host=127.0.0.1 --port=$(PORT) --ignore='$(BUILDDIR)/*' $(SOURCEDIR) $(BUILDDIR)/html
