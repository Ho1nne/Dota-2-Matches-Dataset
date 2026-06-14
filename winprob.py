"""
winprob.py — кривая вероятности победы по ходу игры (live win probability)
==========================================================================

ИДЕЯ (это и есть то, что показывают на трансляциях Dota):
  В любой момент матча мы хотим оценить вероятность победы Radiant. Чем дальше
  игра, тем точнее прогноз — на 5-й минуте всё почти равно, а на 25-й сильный
  перевес почти всегда означает победу.

КАК ЭТО УСТРОЕНО (просто):
  1. Для каждой минуты M берём «снимок» игры — перевес Radiant над Dire по
     нетворту, опыту, добиваниям и убийствам на минуте M.
  2. Для КАЖДОЙ минуты обучаем свою логистическую регрессию: вход — снимок на
     минуте M, выход — вероятность победы Radiant.
  3. Качество каждой модели меряем по кросс-валидации (ROC-AUC) и строим график
     «предсказуемость по ходу игры» — он должен расти от ~0.65 к ~0.9.
  4. Для одного матча-примера прогоняем его снимок на каждой минуте через модель
     этой минуты — получаем КРИВУЮ вероятности победы.

Почему логистическая регрессия:
  Она сразу выдаёт вероятность (от 0 до 1), а не просто «да/нет». Именно
  вероятность нам и нужна для кривой.

Запуск:
    python collect_matches.py 2000   # сначала скачать матчи (если ещё нет)
    python winprob.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ───── Настройки ──────────────────────────────────────────────
# на каких минутах строим снимок и обучаем модель
MINUTES = list(range(2, 31, 2))      # 2, 4, 6, ... 30
MIN_GAMES = 100                      # минимум матчей, чтобы обучать модель на минуте
RANDOM_STATE = 42

# признаки снимка = перевес Radiant над Dire (положительное → перевес Radiant)
FEATURES = ["networth_adv", "xp_adv", "lh_adv", "kills_adv"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.join(BASE_DIR, "matches")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def savefig(fig: plt.Figure, name: str) -> None:
    """Сохраняет график в figures/ и закрывает его."""
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [график] {os.path.relpath(path)}")


# ───── 1. Снимок игры на конкретной минуте ────────────────────

def kills_before(player: dict, seconds: int) -> int:
    """Сколько убийств игрок успел сделать до момента `seconds`."""
    kills_log = player.get("kills_log") or []
    return sum(1 for k in kills_log if k.get("time", 1e9) <= seconds)


def snapshot_at(match: dict, minute: int) -> dict | None:
    """Снимок матча на `minute`-й минуте: перевес Radiant над Dire.

    Возвращает словарь с признаками или None, если до этой минуты игра не дошла
    (или у матча нет поминутных данных).
    """
    players = match.get("players") or []
    if len(players) != 10:
        return None

    seconds = minute * 60
    radiant = {"networth": 0, "xp": 0, "lh": 0, "kills": 0}
    dire = {"networth": 0, "xp": 0, "lh": 0, "kills": 0}

    for p in players:
        gold_t = p.get("gold_t") or []      # нетворт по минутам
        xp_t = p.get("xp_t") or []          # опыт по минутам
        lh_t = p.get("lh_t") or []          # добивания по минутам

        # если массивы короче — значит игра закончилась раньше этой минуты
        if len(gold_t) <= minute or len(xp_t) <= minute:
            return None

        side = radiant if p.get("isRadiant") else dire
        side["networth"] += gold_t[minute]
        side["xp"] += xp_t[minute]
        side["lh"] += lh_t[minute] if len(lh_t) > minute else 0
        side["kills"] += kills_before(p, seconds)

    return {
        "networth_adv": radiant["networth"] - dire["networth"],
        "xp_adv": radiant["xp"] - dire["xp"],
        "lh_adv": radiant["lh"] - dire["lh"],
        "kills_adv": radiant["kills"] - dire["kills"],
    }


# ───── 2. Сбор данных по всем минутам за один проход ───────────

def build_snapshots() -> dict:
    """Читает все матчи один раз и собирает снимки для каждой минуты.

    Возвращает словарь: минута → {"X": матрица признаков, "y": исходы}.
    """
    files = glob.glob(os.path.join(MATCH_DIR, "*.json"))
    if not files:
        raise SystemExit("Нет данных в matches/. Запусти: python collect_matches.py")

    data = {m: {"X": [], "y": []} for m in MINUTES}

    for fp in files:
        with open(fp) as f:
            match = json.load(f)
        if match.get("radiant_win") is None:
            continue
        win = int(bool(match["radiant_win"]))      # 1 — победил Radiant, 0 — Dire

        for minute in MINUTES:
            snap = snapshot_at(match, minute)
            if snap is not None:                    # матч дошёл до этой минуты
                data[minute]["X"].append([snap[f] for f in FEATURES])
                data[minute]["y"].append(win)

    print(f"Прочитано матчей: {len(files)}")
    return data


# ───── 3. Обучение модели на каждой минуте ────────────────────

def make_model():
    """Логистическая регрессия со стандартизацией — выдаёт вероятность победы."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def train_per_minute(data: dict) -> dict:
    """Для каждой минуты: меряет AUC по кросс-валидации и обучает модель на всех.

    Возвращает словарь: минута → {"model": обученная модель, "auc": ..., "n": ...}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    result = {}

    print(f"\n{'минута':>7}{'матчей':>9}{'ROC-AUC':>10}")
    for minute in MINUTES:
        X = np.array(data[minute]["X"], dtype=float)
        y = np.array(data[minute]["y"])

        # пропускаем минуту, если матчей мало или нет обоих исходов
        if len(y) < MIN_GAMES or len(np.unique(y)) < 2:
            continue

        auc = cross_val_score(make_model(), X, y, cv=cv, scoring="roc_auc").mean()
        model = make_model().fit(X, y)            # финальная модель на всех данных
        result[minute] = {"model": model, "auc": auc, "n": len(y)}
        print(f"{minute:>7}{len(y):>9}{auc:>10.3f}")

    return result


# ───── 4. График: предсказуемость по ходу игры ────────────────

def plot_auc_by_minute(result: dict) -> None:
    """Как растёт точность прогноза с течением матча."""
    minutes = sorted(result)
    aucs = [result[m]["auc"] for m in minutes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(minutes, aucs, marker="o")
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="случайное угадывание")
    ax.set(title="Предсказуемость победы по ходу игры",
           xlabel="минута матча", ylabel="ROC-AUC (точность прогноза)",
           ylim=(0.5, 1.0))
    ax.legend()
    savefig(fig, "13_winprob_auc_by_minute.png")


# ───── 5. Кривая вероятности победы для одного матча ──────────

def plot_winprob_curve(result: dict) -> None:
    """Берёт матч-пример и рисует, как менялась вероятность победы Radiant.

    Пример (файл 8823581121) лежит в корне репозитория и НЕ входит в обучение
    (модели учатся на папке matches/). Это иллюстрация работы модели.
    """
    example = os.path.join(BASE_DIR, "8823581121")
    if not os.path.exists(example):
        print("\n(Файл-пример 8823581121 не найден — кривую пропускаем.)")
        return
    with open(example) as f:
        match = json.load(f)

    minutes, probs = [], []
    for minute in sorted(result):
        snap = snapshot_at(match, minute)
        if snap is None:                           # матч уже закончился
            break
        X_one = np.array([[snap[f] for f in FEATURES]], dtype=float)
        p_radiant = result[minute]["model"].predict_proba(X_one)[0, 1]
        minutes.append(minute)
        probs.append(p_radiant)

    actual_win = bool(match.get("radiant_win"))
    winner = "Radiant" if actual_win else "Dire"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(minutes, probs, marker="o", color="#2980b9")
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.fill_between(minutes, 0.5, probs, alpha=0.15, color="#2980b9")
    ax.set(title=f"Вероятность победы Radiant по ходу матча\n"
                 f"(на самом деле победил: {winner})",
           xlabel="минута матча", ylabel="P(победы Radiant)", ylim=(0, 1))
    savefig(fig, "14_winprob_curve_example.png")

    print(f"\nКривая для матча-примера (победил {winner}):")
    for m, p in zip(minutes, probs):
        print(f"  {m:>2} мин: P(Radiant) = {p:.0%}")


# ───── main ───────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Кривая вероятности победы по ходу игры")
    print("=" * 60)

    data = build_snapshots()
    result = train_per_minute(data)

    plot_auc_by_minute(result)
    plot_winprob_curve(result)

    print(f"\nГрафики сохранены в: {os.path.relpath(FIG_DIR)}/")


if __name__ == "__main__":
    main()
