# Telco Churn — быстрый запуск


## Требования
- Python ≥ 3.10, установлен `make`
- Данные: `data/train.csv` (путь и таргет заданы в `configs/default.yaml`)

## 1) Клонирование репозитория
```bash
git clone https://github.com/EugenePanfilov/Telco-Customer-Churn.git
cd telco-customer-churn
```

## 2) Установка (создаст `.venv` и поставит зависимости)
```bash
make install
```

## 3) Обучение
```bash
make train
```

## 4) Оценка
```bash
make eval
```

## 5) Тесты
```bash
make test
```

## 6) Очистка артефактов
```bash
make clean
```

---


## Коротко для Google Colab
```bash
!git clone https://github.com/EugenePanfilov/Telco-Customer-Churn.git
%cd Telco-Customer-Churn
!make install
!make train
!make eval
```

> Артефакты и отчёты: `artifacts/runs/<YYYYmmdd-HHMMSS>/` и симлинк на последний запуск `artifacts/latest/`.
