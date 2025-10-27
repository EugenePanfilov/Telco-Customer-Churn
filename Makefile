# Makefile (устойчивый к python/python3)
PYTHON ?= $(shell command -v python || command -v python3)
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
PIP    := $(BIN)/pip

.PHONY: venv install train eval clean

# 1) создать виртуальное окружение .venv
venv:
	$(PYTHON) -m venv $(VENV)

# 2) установить зависимости в .venv
install: venv
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

# 3) обучение по configs/default.yaml
train:
	$(PY) -m src.train --config configs/default.yaml

# 4) оценка по configs/default.yaml
eval:
	$(PY) -m src.evaluate --config configs/default.yaml

# удалить артефакты
clean:
	rm -rf artifacts