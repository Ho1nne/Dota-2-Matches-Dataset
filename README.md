<div align="center">

# 🎮 Dota 2 Pro Matches — EDA & Win Prediction

**Разведочный анализ профессиональных матчей Dota 2, сравнение ML-моделей и кривая вероятности победы по ходу игры**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)](https://matplotlib.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ektarr/dota-2-pro-matches)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

<table>
<tr>
<td><img src="figures/13_winprob_auc_by_minute.png" width="100%"></td>
<td><img src="figures/14_winprob_curve_example.png" width="100%"></td>
</tr>
</table>

</div>

> [!TIP]
> **Главное за 10 секунд**
> - 📊 **EDA** профессиональных матчей: грейн данных, пропуски, тренды, баланс таргета.
> - 🤖 **Сравнение моделей** на 10-й минуте: логистическая регрессия (**ROC-AUC 0.752**) обходит градиентный бустинг — связь «перевес → победа» почти линейная.
> - 📈 **Win-probability**: предсказуемость растёт по ходу матча с **0.63** (2 мин) до **0.91** (30 мин).

---

## 📑 Содержание

- [О проекте](#-о-проекте)
- [Данные](#-данные)
- [Что делает код](#-что-делает-код)
- [Результаты EDA](#-результаты-eda)
- [Сравнение моделей (10-я минута)](#-сравнение-моделей-10-я-минута)
- [Кривая вероятности победы](#-кривая-вероятности-победы)
- [Структура репозитория](#-структура-репозитория)
- [Запуск](#-запуск)
- [Источники данных](#-источники-данных)

---

## 🎯 О проекте

Проект из трёх логических частей:

| Часть | Что делаем |
| --- | --- |
| **1. EDA** | Изучаем структуру датасета, качество данных (пропуски, типы), распределения, временные тренды и баланс таргета. |
| **2. Сравнение моделей** | По состоянию игры на 10-й минуте предсказываем победителя карты; сравниваем логрегрессию и градиентный бустинг на кросс-валидации. |
| **3. Win-probability** | Обобщаем прогноз на весь матч и строим кривую вероятности победы по минутам — как на турнирных трансляциях. |

Детальная статистика каждой карты берётся напрямую из [OpenDota API](https://docs.opendota.com/) — драфт (picks/bans), поминутные графики золота/опыта, статистика игроков.

---

## 🗃️ Данные

Базовый датасет: **[Dota 2 Pro Matches](https://www.kaggle.com/datasets/ektarr/dota-2-pro-matches)** (Kaggle), загружается автоматически через [`kagglehub`](https://github.com/Kaggle/kagglehub):

| Файл | Содержание |
| --- | --- |
| `tier1_games.csv` | Матчи топового уровня (Tier 1) — основа EDA |
| `teams.csv` | Профессиональные команды |
| `tournaments.csv` | Турниры и лиги |
| `players.csv` | Игроки |

Для моделей по `dota_game_id` дополнительно скачивается **полный JSON** каждой карты из OpenDota: `picks_bans`, `radiant_win`, поминутные массивы `gold_t` / `xp_t` / `lh_t` у игроков.

---

## ⚙️ Что делает код

| Скрипт | Назначение |
| --- | --- |
| [`eda.py`](eda.py) | EDA по `tier1_games.csv`: очистка, пропуски, распределения, тренды, графики |
| [`collect_matches.py`](collect_matches.py) | Массовая выгрузка детальной статистики матчей из OpenDota в `matches/` |
| [`model.py`](model.py) | Признаки на 10-й минуте → **сравнение моделей** (логрегрессия vs бустинг) на CV |
| [`winprob.py`](winprob.py) | **Кривая вероятности победы**: своя модель на каждой минуте матча |

---

## 📊 Результаты EDA

> [!NOTE]
> **Грейн данных:** «1 строка = 1 карта внутри серии». Серийные поля (`team1_win`, `score`, `bestOf`) повторяются по строкам-картам, поэтому для серийной аналитики строки дедуплицируются по `match_id`.

<table>
<tr>
<td width="50%"><img src="figures/01_missing_values.png" width="100%"><br><sub><b>Пропуски в данных</b></sub></td>
<td width="50%"><img src="figures/02_series_format.png" width="100%"><br><sub><b>Формат серий (Bo1/Bo3/Bo5)</b></sub></td>
</tr>
<tr>
<td width="50%"><img src="figures/03_series_per_year.png" width="100%"><br><sub><b>Серии по годам</b></sub></td>
<td width="50%"><img src="figures/06_tournaments_coverage.png" width="100%"><br><sub><b>Турниры и покрытие статистикой</b></sub></td>
</tr>
<tr>
<td width="50%"><img src="figures/04_target_balance.png" width="100%"><br><sub><b>Баланс таргета</b></sub></td>
<td width="50%"><img src="figures/05_teams.png" width="100%"><br><sub><b>Топ команд</b></sub></td>
</tr>
</table>

Доля карт с детальной статистикой (`has_game_data`) выросла с ~0 до 2017 года до ~100 % к последним сезонам. Доля побед `team1` заметно выше 0.5 и зависит от формата — значит, **порядок `team1`/`team2` несёт информацию** (вероятно, `team1` — фаворит). Поэтому модели учатся на нейтральном таргете `radiant_win`, а не на `team1_win`.

---

## 🤖 Сравнение моделей (10-я минута)

**Задача:** по состоянию игры на 10-й минуте предсказать, победит ли Radiant (`radiant_win = 1`).

**Признаки** — перевес Radiant над Dire на 10-й минуте: нетворт, опыт, добивания (LH), денаи (DN), убийства, суммарные уровни, разрушенные башни, первая кровь.

**Сравнение на 5-блочной кросс-валидации** (плюс простое правило для ориентира):

| Подход | Accuracy | ROC-AUC |
| --- | :---: | :---: |
| Базовое правило (`нетворт > 0`) | 0.672 | — |
| Градиентный бустинг (`HistGradientBoosting`) | 0.675 | 0.734 |
| 🏆 **Логистическая регрессия** | **0.684** | **0.752** |

<sub>выборка ~1900 карт; метрики — среднее по 5 фолдам</sub>

Лучшей оказывается **логистическая регрессия**: связь «перевес → победа» почти линейная, поэтому бустинг не даёт преимущества.

<table>
<tr>
<td width="50%"><img src="figures/07_roc_curve.png" width="100%"><br><sub><b>ROC-кривая</b></sub></td>
<td width="50%"><img src="figures/08_feature_importance.png" width="100%"><br><sub><b>Важность признаков (permutation)</b></sub></td>
</tr>
</table>

Сильнее всего влияют **нетворт** и **опыт**; башни/первая кровь/денаи почти не добавляют сигнала.

---

## 📈 Кривая вероятности победы

[`winprob.py`](winprob.py) обобщает идею на **весь матч**: для каждой минуты строится свой «снимок» игры и обучается своя логистическая регрессия. Так на любой минуте можно выдать вероятность победы Radiant.

**Предсказуемость растёт по ходу игры** — ROC-AUC по минутам:

| Минута | 2 | 10 | 20 | 30 |
| --- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.63 | 0.76 | 0.86 | 0.91 |

На 2-й минуте исход почти не определён, к 30-й — предсказуем уверенно. Кривая для отложенного матча-примера сходится к фактическому победителю (см. графики в начале README).

> [!WARNING]
> На поздних минутах матчей меньше (короткие игры до них не доходят) — это *selection bias*. Он виден в колонке «матчей» при запуске `winprob.py`.

---

## 🗂️ Структура репозитория

```
Dota-2-Matches-Dataset/
├── eda.py               # EDA: очистка, анализ, графики
├── collect_matches.py   # Массовая выгрузка матчей в matches/ (для моделей)
├── model.py             # 10-я минута: сравнение моделей (логрег vs бустинг)
├── winprob.py           # Кривая вероятности победы по ходу игры
├── requirements.txt     # Зависимости проекта
├── figures/             # Сохранённые графики (.png)
├── matches/             # Скачанные матчи (JSON, не в гите — создаётся скриптом)
├── 8823581121           # Пример матча из OpenDota (для демо-прогнозов)
└── README.md
```

---

## 🚀 Запуск

```bash
# 1. Окружение и зависимости
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. EDA: отчёт в консоль + графики в figures/
python eda.py

# 3. Скачать матчи для моделей (один раз; качается в matches/)
python collect_matches.py 2000   # ~2000 карт через OpenDota API

# 4. Сравнение моделей на 10-й минуте
python model.py

# 5. Кривая вероятности победы по ходу игры
python winprob.py
```

---

## 🔗 Источники данных

- Датасет: [Dota 2 Pro Matches — Kaggle](https://www.kaggle.com/datasets/ektarr/dota-2-pro-matches)
- API: [OpenDota API](https://docs.opendota.com/)

---

<div align="center">

⭐ **Если проект оказался полезен — поставьте звезду!**

</div>
