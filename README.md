<div align="center">

# Dota 2 Pro Matches — EDA

**Разведочный анализ профессиональных матчей Dota 2**

Загрузка датасета с Kaggle, анализ пропусков, визуализации и выгрузка детальной статистики матчей через OpenDota API.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## Содержание

- [Dota 2 Pro Matches — EDA](#dota-2-pro-matches--eda)
  - [Содержание](#содержание)
  - [О проекте](#о-проекте)
  - [Данные](#данные)
  - [Что делает код](#что-делает-код)
  - [Результаты EDA](#результаты-eda)
  - [Структура репозитория](#структура-репозитория)
  - [Запуск](#запуск)
  - [Источники данных](#источники-данных)


---

## О проекте

Проект посвящён разведочному анализу данных (EDA) по профессиональным матчам **Dota 2**.
Цель — изучить структуру датасета, оценить качество данных (пропуски, типы, распределения)
и подготовить почву для дальнейшего анализа и моделирования.

Дополнительно проект показывает, как обогатить табличные данные **детальной статистикой
конкретного матча** напрямую из [OpenDota API](https://docs.opendota.com/) — вплоть до
тимфайтов, драфта (picks/bans), графиков преимущества по золоту и опыту.

---

## Данные

Базовый датасет: **[Dota 2 Pro Matches](https://www.kaggle.com/datasets/ektarr/dota-2-pro-matches)** (Kaggle).

Загружается автоматически через [`kagglehub`](https://github.com/Kaggle/kagglehub) и состоит из нескольких таблиц:

| Файл | Содержание |
| --- | --- |
| `teams.csv` | Профессиональные команды |
| `tournaments.csv` | Турниры и лиги |
| `players.csv` | Игроки |
| `tier1_games.csv` | Матчи топового уровня (Tier 1) |

Для выбранного матча (`dota_game_id`) дополнительно скачивается **полный JSON** из OpenDota API
с подробной статистикой: `teamfights`, `objectives`, `picks_bans`, `radiant_gold_adv`,
`radiant_xp_adv`, `players` и многое другое.

---

## Что делает код

Основной анализ — в [`eda.py`](eda.py) (фокус на `tier1_games.csv`):

-  **Загружает** датасет с Kaggle через `kagglehub`
-  **Объясняет грейн данных**: строка = одна *карта* внутри *серии* (`match_id`)
-  **Чистит** данные: даты-нули `1970-01-01` → `NaT`, аномальный `bestOf` (4/6/7)
-  **Анализирует** пропуски, распределения, временные тренды
-  **Считает** баланс таргета, топ команд и турниров
-  **Сохраняет** графики в `figures/` и печатает текстовый отчёт

Скрипт [`test.py`](test.py) дополнительно **выгружает детальную статистику матча**
по `dota_game_id` из OpenDota API и сохраняет её в JSON.

---

## Результаты EDA

> Данные: **`tier1_games.csv`** — 48 931 карта → **29 059 серий** (≈1.68 карты на серию).
> Серийные поля (`team1_win`, `score`, `bestOf`) повторяются по строкам-картам,
> поэтому для серийной аналитики строки дедуплицируются по `match_id`.

**Пропуски в данных**

![Пропуски](figures/01_missing_values.png)

`dota_game_id` отсутствует у 28 % карт, `game_id` — у 23 %, ID игроков — 3.5–5.8 %.

**Формат серий и число карт**

![Формат серий](figures/02_series_format.png)

Преобладают Bo3 и Bo1; Bo5 — финалы и решающие матчи.

**Динамика по годам и покрытие детальной статистикой**

![Серии по годам](figures/03_series_per_year.png)

![Турниры и покрытие](figures/06_tournaments_coverage.png)

Доля карт с детальной статистикой (`has_game_data`) выросла с ~0 до 2017 года
до ~100 % к 2023 — у старых матчей подробных данных OpenDota нет.

**Баланс таргета `team1_win`**

![Баланс таргета](figures/04_target_balance.png)

Средняя доля побед `team1` = **0.525**, но сильно зависит от формата (Bo5 → 0.625).
Перекос от 0.5 означает, что **порядок `team1`/`team2` несёт информацию** (вероятно,
`team1` — фаворит/верхняя сетка) — это потенциальная утечка при моделировании.

**Команды**

![Команды](figures/05_teams.png)

По объёму матчей лидируют Team Liquid, Virtus.pro и Natus Vincere;
по винрейту (среди команд с ≥50 серий) — Team NP, Team Falcons и Team Secret.

---

## Структура репозитория

```
Dota-2-Matches-Dataset/
├── eda.py             # Основной EDA-скрипт: очистка, анализ, графики
├── test.py            # Загрузка данных + выгрузка матча из OpenDota API
├── requirements.txt   # Зависимости проекта
├── figures/           # Сохранённые графики EDA (.png)
├── 8823581121         # Пример выгруженного матча (JSON из OpenDota API)
└── README.md
```

---

## Запуск

```bash
# 1. Окружение и зависимости
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. EDA: отчёт в консоль + графики в figures/
python eda.py
```

Чтобы выгрузить детальную статистику другого матча, поменяйте ID в [`test.py`](test.py):

```python
dota_game_id = 8823581121  # ← подставьте нужный ID из колонки dota_game_id
```

---

## Источники данных

- Датасет: [Dota 2 Pro Matches — Kaggle](https://www.kaggle.com/datasets/ektarr/dota-2-pro-matches)
- API: [OpenDota API](https://docs.opendota.com/)

---

<div align="center">

Если проект оказался полезен — поставьте звезду!

</div>
