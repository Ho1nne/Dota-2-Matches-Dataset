"""
EDA — Dota 2 Pro Matches (Tier 1)
=================================

Разведочный анализ таблицы `tier1_games.csv` из датасета
"Dota 2 Pro Matches" (Kaggle, ektarr/dota-2-pro-matches).

Что делает скрипт:
  1. Скачивает датасет через kagglehub и загружает tier1_games.csv
  2. Объясняет грейн данных (строка = отдельная карта внутри серии)
  3. Чистит данные (даты-нули 1970, аномальный bestOf, типы)
  4. Анализирует пропуски
  5. Считает распределения, временные тренды, баланс таргета,
     топ команд и турниров
  6. Сохраняет графики в ./figures и печатает текстовый отчёт

Запуск:
    python eda.py
"""

from __future__ import annotations

import os
import textwrap

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ───── Настройки отображения ──────────────────────────────────
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid", palette="deep")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Серийные поля повторяются на каждой строке-карте → дедуп до уровня серии
SERIES_COLS = [
    "match_id",
    "tournament_id",
    "tournament_en",
    "team1_id",
    "team1",
    "team2_id",
    "team2",
    "score1",
    "score2",
    "bestOf",
    "bestOf_clean",
    "datetime",
    "year",
    "month",
    "team1_win",
    "games_played",
]


def header(title: str) -> None:
    """Печатает заголовок секции отчёта."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def savefig(fig: plt.Figure, name: str) -> None:
    """Сохраняет фигуру в ./figures и закрывает её."""
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {os.path.relpath(path)}")


# ───── 1. Загрузка ────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    header("1. ЗАГРУЗКА ДАННЫХ")
    path = kagglehub.dataset_download("ektarr/dota-2-pro-matches")
    print(f"Источник: {path}")
    df = pd.read_csv(os.path.join(path, "tier1_games.csv"))
    print(f"tier1_games.csv  →  {df.shape[0]:,} строк × {df.shape[1]} колонок")
    return df


# ───── 2. Обзор структуры ─────────────────────────────────────


def overview(df: pd.DataFrame) -> None:
    header("2. ОБЗОР СТРУКТУРЫ")
    print(
        textwrap.dedent("""\
        Грейн данных: 1 строка = 1 КАРТА (game) внутри СЕРИИ (match).
          match_id      — id серии (Bo1/Bo3/Bo5)
          game_id       — id отдельной карты внутри серии
          dota_game_id  — id карты в OpenDota
        Серийные поля (score1/2, games_played, team1_win, bestOf)
        дублируются на каждой строке-карте серии.
    """)
    )
    print("Типы данных:")
    print(df.dtypes.to_string())
    print("\nПример строк:")
    show = [
        "match_id",
        "game_id",
        "tournament_en",
        "team1",
        "team2",
        "score1",
        "score2",
        "bestOf",
        "team1_win",
        "datetime",
    ]
    print(df[show].head().to_string())


# ───── 3. Очистка ─────────────────────────────────────────────


def clean(df: pd.DataFrame) -> pd.DataFrame:
    header("3. ОЧИСТКА ДАННЫХ")
    df = df.copy()

    # datetime: нули-эпоха (1970-01-01) — это отсутствующая дата
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    epoch_bad = dt <= pd.Timestamp("1971-01-01")
    n_bad = int(epoch_bad.sum() + dt.isna().sum())
    dt = dt.mask(epoch_bad)
    df["datetime"] = dt
    df["year"] = dt.dt.year
    df["month"] = dt.dt.to_period("M").astype("string")
    print(f"datetime: {n_bad} некорректных дат (1970/NaT) → NaT")

    # bestOf: валидны 1,2,3,5; остальное (4,6,7) — аномалии
    valid_bo = {1, 2, 3, 5}
    bad_bo = ~df["bestOf"].isin(valid_bo) & df["bestOf"].notna()
    print(
        f"bestOf: {int(bad_bo.sum())} аномальных значений "
        f"({sorted(df.loc[bad_bo, 'bestOf'].dropna().unique().tolist())})"
    )
    df["bestOf_clean"] = df["bestOf"].where(~bad_bo)

    # id-колонки игроков/игр читаются как float — это ожидаемо из-за NaN
    print("Идентификаторы хранятся как float (есть пропуски) — оставляем как есть.")
    return df


# ───── 4. Пропуски ────────────────────────────────────────────


def missing_values(df: pd.DataFrame) -> None:
    header("4. АНАЛИЗ ПРОПУСКОВ")
    miss = (df.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss.to_string() if len(miss) else "Пропусков нет.")

    if len(miss):
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(miss))))
        sns.barplot(
            x=miss.values,
            y=miss.index,
            hue=miss.index,
            palette="rocket",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("% пропусков")
        ax.set_title("Доля пропущенных значений по колонкам (Tier 1)")
        for i, v in enumerate(miss.values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
        savefig(fig, "01_missing_values.png")


# ───── 5. Серии vs карты ──────────────────────────────────────


def series_analysis(df: pd.DataFrame) -> pd.DataFrame:
    header("5. СЕРИИ vs КАРТЫ")
    series = df.drop_duplicates(subset="match_id")[SERIES_COLS].copy()
    print(f"Карт (строк):  {len(df):,}")
    print(f"Серий (match): {len(series):,}")
    print(f"В среднем карт на серию: {len(df) / len(series):.2f}")

    cards_per = df.groupby("match_id").size()
    print("\nРаспределение числа карт в серии:")
    print(cards_per.value_counts().sort_index().to_string())

    # Распределение bestOf на уровне серий
    bo = series["bestOf_clean"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(
        x=cards_per.value_counts().sort_index().index,
        y=cards_per.value_counts().sort_index().values,
        hue=cards_per.value_counts().sort_index().index,
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set(title="Число карт в серии", xlabel="карт", ylabel="серий")
    sns.barplot(
        x=bo.index.astype(int).astype(str),
        y=bo.values,
        hue=bo.index,
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="Формат серий (bestOf)", xlabel="Best of", ylabel="серий")
    savefig(fig, "02_series_format.png")
    return series


# ───── 6. Временные тренды ────────────────────────────────────


def temporal(series: pd.DataFrame) -> None:
    header("6. ВРЕМЕННЫЕ ТРЕНДЫ")
    yr = series.dropna(subset=["year"])
    by_year = yr["year"].astype(int).value_counts().sort_index()
    print("Серий по годам:")
    print(by_year.to_string())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(x=by_year.index, y=by_year.values, marker="o", ax=ax)
    ax.set(title="Tier 1: число серий по годам", xlabel="год", ylabel="серий")
    savefig(fig, "03_series_per_year.png")


# ───── 7. Баланс таргета ──────────────────────────────────────


def target_balance(series: pd.DataFrame) -> None:
    header("7. БАЛАНС ТАРГЕТА (team1_win)")
    vc = series["team1_win"].value_counts(dropna=False)
    print(vc.to_string())
    print(f"\nДоля побед team1: {series['team1_win'].mean():.3f}")
    print(
        "(заметный перекос от 0.5 = порядок team1/team2 несёт информацию,"
        " напр. team1 — фаворит/верхняя сетка)"
    )

    by_bo = series.groupby("bestOf_clean")["team1_win"].mean()
    print("\nДоля побед team1 по формату:")
    print(by_bo.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        x=by_bo.index.astype(int).astype(str),
        y=by_bo.values,
        hue=by_bo.index,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.axhline(0.5, ls="--", c="red", lw=1)
    ax.set(
        title="Доля побед team1 по bestOf",
        xlabel="Best of",
        ylabel="P(team1_win)",
        ylim=(0, 1),
    )
    savefig(fig, "04_target_balance.png")


# ───── 8. Команды ─────────────────────────────────────────────


def teams_analysis(series: pd.DataFrame) -> None:
    header("8. КОМАНДЫ")
    # «Длинная» таблица: одна строка на участие команды в серии
    t1 = series[["team1", "team1_win"]].rename(columns={"team1": "team"})
    t1["win"] = series["team1_win"]
    t2 = series[["team2", "team1_win"]].rename(columns={"team2": "team"})
    t2["win"] = 1 - series["team1_win"]
    long = pd.concat([t1[["team", "win"]], t2[["team", "win"]]], ignore_index=True)
    long = long.dropna(subset=["team"])

    agg = long.groupby("team").agg(
        series_played=("win", "size"), winrate=("win", "mean")
    )
    top_games = agg.sort_values("series_played", ascending=False).head(15)
    print("Топ-15 команд по числу серий:")
    print(top_games.round(3).to_string())

    # Винрейт среди команд с весомой выборкой
    qualified = agg[agg["series_played"] >= 50].sort_values("winrate", ascending=False)
    print("\nТоп-10 по винрейту (≥50 серий):")
    print(qualified.head(10).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        x=top_games["series_played"],
        y=top_games.index,
        hue=top_games.index,
        palette="mako",
        legend=False,
        ax=axes[0],
    )
    axes[0].set(title="Топ-15 команд по числу серий", xlabel="серий", ylabel="")
    qt = qualified.head(12)
    sns.barplot(
        x=qt["winrate"],
        y=qt.index,
        hue=qt.index,
        palette="rocket",
        legend=False,
        ax=axes[1],
    )
    axes[1].axvline(0.5, ls="--", c="grey", lw=1)
    axes[1].set(title="Топ по винрейту (≥50 серий)", xlabel="winrate", ylabel="")
    savefig(fig, "05_teams.png")


# ───── 9. Турниры и покрытие статистикой ──────────────────────


def tournaments_and_coverage(df: pd.DataFrame, series: pd.DataFrame) -> None:
    header("9. ТУРНИРЫ И ПОКРЫТИЕ ДЕТАЛЬНОЙ СТАТИСТИКОЙ")
    top_t = series["tournament_en"].value_counts().head(15)
    print("Топ-15 турниров по числу серий:")
    print(top_t.to_string())

    # has_game_data — есть ли подробная статистика карты в OpenDota
    cov = df["has_game_data"].mean()
    print(f"\nПокрытие детальной статистикой (has_game_data=1): {cov:.1%} карт")
    by_year = (
        df.dropna(subset=["year"])
        .groupby(df["year"].astype("Int64"))["has_game_data"]
        .mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(
        x=top_t.values,
        y=top_t.index,
        hue=top_t.index,
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set(title="Топ-15 турниров по числу серий", xlabel="серий", ylabel="")
    sns.lineplot(x=by_year.index.astype(int), y=by_year.values, marker="o", ax=axes[1])
    axes[1].set(
        title="Доля карт с детальной статистикой по годам",
        xlabel="год",
        ylabel="has_game_data",
        ylim=(0, 1),
    )
    savefig(fig, "06_tournaments_coverage.png")


# ───── main ───────────────────────────────────────────────────


def main() -> None:
    df = load_data()
    overview(df)
    df = clean(df)
    missing_values(df)
    series = series_analysis(df)
    temporal(series)
    target_balance(series)
    teams_analysis(series)
    tournaments_and_coverage(df, series)

    header("ГОТОВО")
    print(f"Графики сохранены в: {os.path.relpath(FIG_DIR)}/")
    print("Файлы: 01_missing_values, 02_series_format, 03_series_per_year,")
    print("       04_target_balance, 05_teams, 06_tournaments_coverage (.png)")


if __name__ == "__main__":
    main()
