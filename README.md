# Customer Churn Prediction

## Описание

Цель проекта — предсказать вероятность оттока пользователя
подписочного сервиса на основе его характеристик.

Задача формулируется как бинарная классификация:
пользователь либо уйдёт, либо останется.

## Структура проекта

```text
ml-churn-prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── performance/
│   │   ├── __init__.py
│   │   └── test_benchmark.py
│   └── unit/
│       ├── __init__.py
│       ├── test_model.py
│       ├── test_preprocessing.py
│       └── test_validation.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Данные

Используется открытый датасет Telco Customer Churn.
Каждая строка соответствует одному пользователю.
Целевая переменная — `Churn`.

## Подход

Проект реализован как end-to-end ML-пайплайн:

- валидация сырого набора данных;
- очистка и feature engineering (`pd.get_dummies`);
- разделение данных на train/test без утечки;
- обучение baseline-модели (Logistic Regression);
- оценка качества по ROC-AUC.

### Что делает preprocessing

В `src/preprocessing.py` реализованы три шага:

- `validate_raw_data(df)`:
	- проверяет обязательные колонки (`customerID`, `Churn`, `TotalCharges`);
	- отклоняет пустой датасет;
	- отклоняет недопустимые значения таргета (кроме `Yes`/`No`).
- `prepare_features(df)`:
	- приводит `TotalCharges` к числу;
	- заменяет невалидные/пустые `TotalCharges` на `0.0`;
	- кодирует `Churn` в `0/1`;
	- кодирует категориальные признаки через one-hot (`drop_first=True`).
- `load_and_prepare_data(path)`:
	- читает CSV;
	- валидирует вход;
	- возвращает подготовленные `X, y`.

## Установка

```bash
python -m pip install -r requirements.txt
python -m pip install pytest pytest-benchmark
```

Если используешь venv:

```bash
source .venv/bin/activate
```

## Запуск обучения

```bash
python -m src.train
```

## Метрика

В качестве основной метрики используется ROC-AUC,
так как классы несбалансированы,
а задача сводится к ранжированию пользователей по риску ухода.

## Результат

Baseline-модель (логистическая регрессия)
показывает ROC-AUC около `0.84` на тестовой выборке.

## Тестирование

Тесты организованы на `pytest` и покрывают:

- unit-тесты предобработки и feature engineering;
- unit-тесты валидации входных данных и edge-кейсов;
- unit-тесты модели (`predict`, `predict_proba`, ROC-AUC);
- performance-тесты на `pytest-benchmark` (предобработка, feature engineering, инференс).

### Маркеры

Настроены маркеры:

- `smoke` — быстрые критичные проверки;
- `slow` — более долгие тесты;
- `performance` — benchmark-сценарии.

### Команды запуска

Полный прогон:

```bash
python -m pytest -q
```

Только unit:

```bash
python -m pytest tests/unit -q
```

Только performance:

```bash
python -m pytest tests/performance -q
```

Только smoke:

```bash
python -m pytest -m smoke -q
```

Исключить долгие/benchmark:

```bash
python -m pytest -m "not slow and not performance" -q
```

## Интерпретация

Анализ коэффициентов модели показал, что наибольшее влияние
на риск оттока оказывают:

- тип контракта пользователя;
- длительность использования сервиса;
- наличие дополнительных услуг.

Полученные зависимости согласуются с бизнес-логикой.

## Идеи для улучшения

- вынести preprocessing в sklearn `Pipeline`/`ColumnTransformer`;
- добавить кросс-валидацию и model selection;
- учесть дисбаланс через `class_weight` и threshold tuning;
- добавить integration-тесты для полного train-пайплайна;
- подключить CI (GitHub Actions) с запуском `pytest`.
