# Makefile — единый для локали и Colab (всегда .venv)
PYTHON ?= $(shell command -v python || command -v python3)

VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
PIP    := $(BIN)/pip

.PHONY: install train eval test clean

# 1) создать .venv и поставить зависимости
install:
	# создаём venv (если не вышло — пробуем virtualenv)
	$(PYTHON) -m venv $(VENV) || ( $(PYTHON) -m pip install -q virtualenv && $(PYTHON) -m virtualenv -p $(PYTHON) $(VENV) )
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

# 2) обучение по configs/default.yaml
train:
	$(PY) -m src.train --config configs/default.yaml

# 3) оценка по configs/default.yaml
eval:
	$(PY) -m src.evaluate --config configs/default.yaml

# 4) тесты (нужно, чтобы pytest был в requirements.txt)
test:
	$(PY) -m pytest -q

# очистка артефактов
clean:
	rm -rf artifacts