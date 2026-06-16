"""
winprob_gif.py — анимированная кривая вероятности победы (GIF)
=============================================================

Делает «живую» версию графика из winprob.py: для матча-примера кривая
вероятности победы Radiant вырастает минута за минутой, как live win-probability
на турнирных трансляциях. Справа — полоска текущего расклада (Radiant/Dire).

Переиспользует winprob.py: снимки игры (`snapshot_at`) и модели на каждой
минуте (`build_snapshots` + `train_per_minute`). Сам ничего нового не считает —
только рисует анимацию.

Результат: figures/winprob.gif

Запуск:
    python collect_matches.py 2000   # если матчи ещё не скачаны
    python winprob_gif.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from winprob import (
    BASE_DIR,
    FEATURES,
    FIG_DIR,
    build_snapshots,
    snapshot_at,
    train_per_minute,
)

RADIANT_COLOR = "#27ae60"   # зелёный — Radiant
DIRE_COLOR = "#c0392b"      # красный — Dire
FPS = 2.5                   # кадров в секунду
HOLD_FRAMES = 6             # сколько кадров «подержать» финал


def example_curve(result: dict):
    """Считает вероятность победы Radiant по минутам для матча-примера."""
    with open(os.path.join(BASE_DIR, "8823581121")) as f:
        match = json.load(f)

    minutes, probs = [], []
    for minute in sorted(result):
        snap = snapshot_at(match, minute)
        if snap is None:                       # матч уже закончился
            break
        X_one = np.array([[snap[f] for f in FEATURES]], dtype=float)
        probs.append(result[minute]["model"].predict_proba(X_one)[0, 1])
        minutes.append(minute)

    winner = "Radiant" if match.get("radiant_win") else "Dire"
    return minutes, probs, winner


def make_gif() -> None:
    # 1. обучаем модели на каждой минуте (как в winprob.py) и берём пример
    data = build_snapshots()
    result = train_per_minute(data)
    minutes, probs, winner = example_curve(result)

    # 2. готовим фигуру: слева кривая, справа полоска текущего расклада
    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [6, 1]}
    )
    fig.suptitle(f"Вероятность победы Radiant по ходу матча  (победил: {winner})",
                 fontsize=13, fontweight="bold")

    def draw(frame: int) -> None:
        i = min(frame, len(minutes) - 1)        # последние кадры держат финал
        mm, pp = minutes[:i + 1], probs[:i + 1]
        p = pp[-1]
        color = RADIANT_COLOR if p >= 0.5 else DIRE_COLOR

        # --- левая панель: растущая кривая ---
        ax.clear()
        ax.plot(mm, pp, color="#2c3e50", lw=2, zorder=2)
        ax.fill_between(mm, 0.5, pp, alpha=0.15, color=color, zorder=1)
        ax.scatter([mm[-1]], [p], s=120, color=color, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.text(0.03, 0.94, f"{p:.0%}", transform=ax.transAxes,
                fontsize=26, fontweight="bold", color=color, va="top")
        ax.text(0.03, 0.80, f"{mm[-1]}-я минута", transform=ax.transAxes,
                fontsize=12, color="#555", va="top")
        ax.set(xlim=(min(minutes) - 1, max(minutes) + 1), ylim=(0, 1),
               xlabel="минута матча", ylabel="P(победы Radiant)")
        ax.grid(True, alpha=0.3)

        # --- правая панель: расклад в моменте ---
        bar.clear()
        bar.bar(0, p, bottom=1 - p, width=1, color=RADIANT_COLOR)   # Radiant сверху
        bar.bar(0, 1 - p, width=1, color=DIRE_COLOR)               # Dire снизу
        bar.text(0, 1 - p / 2, f"Radiant\n{p:.0%}", ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")
        bar.text(0, (1 - p) / 2, f"Dire\n{1 - p:.0%}", ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")
        bar.set(xlim=(-0.6, 0.6), ylim=(0, 1))
        bar.axis("off")

    frames = len(minutes) + HOLD_FRAMES
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)

    out = os.path.join(FIG_DIR, "winprob.gif")
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"\nГотово: {os.path.relpath(out)}  ({len(minutes)} кадров)")


if __name__ == "__main__":
    make_gif()
