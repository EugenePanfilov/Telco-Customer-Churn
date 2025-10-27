# Commands

```bash
make install   # создать .venv и установить зависимости
make train     # обучить модель:   python -m src.train --config configs/default.yaml
make eval      # оценить модель:   python -m src.evaluate --config configs/default.yaml
make clean     # удалить artifacts/
```