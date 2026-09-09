PYTHON := .venv/bin/python
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
UTILS_EXT := build/local/jitasm/utils$(EXT_SUFFIX)

.PHONY: all test benchmark check

all:
	$(PYTHON) -m build

wheel:
	sudo rm -rf build/ dist/
	sudo CIBW_CONTAINER_ENGINE=podman python -m cibuildwheel --only cp314-manylinux_x86_64 --output-dir dist

$(UTILS_EXT): src/jitasm/utils.c
	mkdir -p build/local/jitasm
	$(CC) -O3 -shared -fPIC -I$(PYTHON_INCLUDE) $< -o $@

test: $(UTILS_EXT)
	PYTHONPATH=build/local:src:tests $(PYTHON) tests/run.py

benchmark: $(UTILS_EXT)
	PYTHONPATH=build/local:src $(PYTHON) benchmark/sqrt.py

