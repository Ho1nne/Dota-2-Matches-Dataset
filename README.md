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

-  **Загружает** датасет с Kaggle через `kagglehub`
-  **Читает** таблицы (`teams`, `tournaments`, `players`, `tier1_games`) в `pandas`
-  **Анализирует** пропущенные значения и структуру данных
-  **Строит** визуализации с помощью `matplotlib`
-  **Выгружает** детальную статистику матча по `dota_game_id` из OpenDota API и сохраняет её в JSON

---

## Структура репозитория

```
Dota-2-Matches-Dataset/
├── test.py        # Основной скрипт: загрузка, анализ, выгрузка матча из OpenDota
├── 8823581121     # Пример выгруженного матча (JSON из OpenDota API)
└── README.md
```

---

## Запуск

```bash
python test.py
```

Скрипт скачает датасет, выведет список файлов, загрузит таблицы и сохранит JSON
выбранного матча в файл с именем, равным `dota_game_id`.

Чтобы проанализировать другой матч, поменяйте значение в `test.py`:

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
