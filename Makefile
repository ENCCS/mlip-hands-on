SPHINXBUILD ?= sphinx-build
AUTOBUILD ?= sphinx-autobuild
PORT ?= 8766

.PHONY: html livehtml
html:
	$(SPHINXBUILD) -n -W -b html content _build/html

livehtml:
	$(AUTOBUILD) -n -W --host=127.0.0.1 --port=$(PORT) --ignore='_build/*' content _build/html
