"""
model.py — прогноз победы по состоянию игры на 10-й минуте
==========================================================

Идея проекта:
  По данным первых 10 минут матча предсказать, какая команда победит.
  В Dota преимущество по золоту и опыту на ранней стадии хорошо
  предсказывает исход, поэтому даже простая модель работает заметно
  лучше угадывания.

Что делает скрипт:
  1. Читает скачанные матчи из папки matches/ (их качает collect_matches.py)
  2. Для каждого матча берёт "срез" на 10-й минуте и считает признаки —
     разница (Radiant − Dire) по: нетворту, опыту, добиваниям, денаям,
     киллам, уровням, разрушенным башням и первой крови
  3. Сравнивает две модели (логрегрессия и градиентный бустинг) на
     5-блочной кросс-валидации и выбирает лучшую по ROC-AUC
  4. Строит графики по честным out-of-fold прогнозам, а важность признаков
     считает через permutation importance (работает для любой модели)
  5. Сохраняет графики в figures/ и печатает отчёт

Откуда берутся признаки:
  В JSON-е OpenDota у каждого игрока есть массивы по минутам:
    gold_t (нетворт), xp_t, lh_t, dn_t  (индекс [10] = 10-я минута).
  Складываем их по игрокам команды и берём разницу команд. Уровень
  восстанавливаем из опыта, башни и первую кровь — из поля objectives.

Запуск:
    python collect_matches.py    # сначала скачать матчи (один раз)
    python model.py              # потом обучить модель
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ───── Настройки ──────────────────────────────────────────────
MINUTE = 10            # на какой минуте делаем срез игры для прогноза
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.join(BASE_DIR, "matches")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

# Признаки модели: разница команд (Radiant − Dire) на 10-й минуте.
# gold_t в OpenDota — это нетворт, поэтому gold_adv = перевес по нетворту.
FEATURES = [
    "gold_adv", "xp_adv", "lh_adv", "dn_adv", "kills_adv",
    "level_adv", "towers_adv", "fb_adv",
]
FEATURE_RU = {
    "gold_adv": "Нетворт",
    "xp_adv": "Опыт",
    "lh_adv": "Добивания (LH)",
    "dn_adv": "Денаи (DN)",
    "kills_adv": "Убийства",
    "level_adv": "Уровни",
    "towers_adv": "Башни",
    "fb_adv": "Первая кровь",
}

# Кумулятивный опыт для достижения уровня (индекс = уровень − 1, до 30).
# Нужен, чтобы из xp_t[10] восстановить уровень героя на 10-й минуте.
XP_TO_LEVEL = [
    0, 230, 600, 1080, 1660, 2260, 2980, 3730, 4510, 5320,
    6160, 7030, 7930, 8865, 9805, 10775, 11775, 12805, 13865, 14955,
    16205, 17605, 19155, 20855, 22705, 24705, 26855, 29155, 31605, 34205,
]


def savefig(fig: plt.Figure, name: str) -> None:
    """Сохраняет график в figures/ и закрывает его."""
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [график] {os.path.relpath(path)}")


# ───── 1. Признаки на 10-й минуте ─────────────────────────────

def kills_before(player: dict, seconds: int) -> int:
    """Сколько убийств игрок сделал до момента `seconds`."""
    log = player.get("kills_log") or []
    return sum(1 for k in log if k.get("time", 1e9) <= seconds)


def xp_to_level(xp: int) -> int:
    """Восстанавливает уровень героя по накопленному опыту (1..30)."""
    lvl = 1
    for i, threshold in enumerate(XP_TO_LEVEL):
        if xp >= threshold:
            lvl = i + 1
        else:
            break
    return lvl


def towers_adv_before(match: dict, seconds: int) -> int:
    """Перевес по разрушенным башням до `seconds` (Radiant − Dire).

    В objectives building_kill.key вида npc_dota_goodguys_tower1_top —
    это башня Radiant (goodguys), значит её снесла Dire → очко Dire.
    """
    r = d = 0
    for o in match.get("objectives") or []:
        if o.get("type") != "building_kill":
            continue
        if (o.get("time") or 1e9) > seconds:
            continue
        key = o.get("key") or ""
        if "tower" not in key:
            continue
        if "goodguys" in key:      # упала башня Radiant → её снесла Dire
            d += 1
        elif "badguys" in key:     # упала башня Dire → её снесла Radiant
            r += 1
    return r - d


def first_blood_adv(match: dict, seconds: int) -> int:
    """Первая кровь до `seconds`: +1 — взяла Radiant, −1 — Dire, 0 — не было."""
    t = match.get("first_blood_time")
    if t is None or t < 0 or t > seconds:
        return 0
    for o in match.get("objectives") or []:
        if o.get("type") == "CHAT_MESSAGE_FIRSTBLOOD":
            ps = o.get("player_slot")
            if ps is None:
                return 0
            return 1 if ps < 128 else -1   # slot < 128 → Radiant
    return 0


def minute_features(match: dict, minute: int = MINUTE) -> dict | None:
    """Считает признаки одного матча на N-й минуте. None, если данных нет."""
    # массивы по минутам есть только у "распарсенных" матчей
    if not match.get("radiant_gold_adv"):
        return None
    players = match.get("players") or []
    if len(players) != 10:
        return None

    sec = minute * 60
    # суммы по двум командам
    agg = {side: dict(gold=0, xp=0, lh=0, dn=0, kills=0, level=0)
           for side in ("radiant", "dire")}

    for p in players:
        gt = p.get("gold_t") or []
        xt = p.get("xp_t") or []
        lt = p.get("lh_t") or []
        dt = p.get("dn_t") or []
        # игра должна длиться хотя бы 10 минут, иначе среза нет
        if len(gt) <= minute or len(xt) <= minute:
            return None

        side = "radiant" if p.get("isRadiant") else "dire"
        agg[side]["gold"] += gt[minute]
        agg[side]["xp"] += xt[minute]
        agg[side]["lh"] += lt[minute] if len(lt) > minute else 0
        agg[side]["dn"] += dt[minute] if len(dt) > minute else 0
        agg[side]["kills"] += kills_before(p, sec)
        agg[side]["level"] += xp_to_level(xt[minute])

    r, d = agg["radiant"], agg["dire"]
    return {
        "match_id": match.get("match_id"),
        # признаки = разница Radiant − Dire (положительное → перевес Radiant)
        "gold_adv": r["gold"] - d["gold"],
        "xp_adv": r["xp"] - d["xp"],
        "lh_adv": r["lh"] - d["lh"],
        "dn_adv": r["dn"] - d["dn"],
        "kills_adv": r["kills"] - d["kills"],
        "level_adv": r["level"] - d["level"],
        "towers_adv": towers_adv_before(match, sec),
        "fb_adv": first_blood_adv(match, sec),
        # что предсказываем: 1 — победил Radiant, 0 — Dire
        "radiant_win": int(bool(match.get("radiant_win"))),
    }


def build_dataset() -> pd.DataFrame:
    """Собирает таблицу признаков из всех матчей в matches/."""
    files = glob.glob(os.path.join(MATCH_DIR, "*.json"))
    if not files:
        raise SystemExit(
            "Нет данных в matches/. Сначала запусти: python collect_matches.py"
        )

    rows, skipped = [], 0
    for fp in files:
        with open(fp) as f:
            match = json.load(f)
        feat = minute_features(match)
        if feat:
            rows.append(feat)
        else:
            skipped += 1

    df = pd.DataFrame(rows)
    print(f"Матчей в папке: {len(files)} | подошло: {len(df)} | "
          f"пропущено (нет графиков/короткие): {skipped}")
    return df


# ───── 2. Обучение и сравнение моделей ────────────────────────

def candidate_models() -> dict:
    """Модели-кандидаты. Логрегрессии нужен StandardScaler, бустингу — нет."""
    return {
        "Логистическая регрессия": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        # регуляризованный, чтобы не переобучаться на ~2k матчей и 8 признаках
        "Градиентный бустинг": HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=40, l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate(df: pd.DataFrame):
    """Сравнивает модели на кросс-валидации и выбирает лучшую по ROC-AUC.

    Возвращает (имя_лучшей, модель, обученная на всех данных).
    """
    X = df[FEATURES]
    y = df["radiant_win"]

    # Стратифицированная 5-блочная CV: каждую модель честно проверяем на 5
    # разных тест-блоках и усредняем — метрики устойчивее, чем при одном сплите.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # простое правило для сравнения: "у кого больше нетворт на 10-й, тот и победит"
    base_acc = accuracy_score(y, (X["gold_adv"] > 0).astype(int))

    print("\n--- Сравнение моделей (5-блочная кросс-валидация) ---")
    print(f"Доля Radiant-побед в данных : {y.mean():.3f}")
    print(f"Базовое правило (нетворт>0) : accuracy = {base_acc:.3f}")

    auc_by_model = {}
    for name, model in candidate_models().items():
        cvres = cross_validate(model, X, y, cv=cv,
                               scoring=["accuracy", "roc_auc"])
        acc, auc = cvres["test_accuracy"], cvres["test_roc_auc"]
        auc_by_model[name] = auc.mean()
        print(f"{name:<24}: accuracy = {acc.mean():.3f} ± {acc.std():.3f} | "
              f"ROC-AUC = {auc.mean():.3f} ± {auc.std():.3f}")

    best_name = max(auc_by_model, key=auc_by_model.get)
    print(f"\nЛучшая модель: {best_name} (ROC-AUC = {auc_by_model[best_name]:.3f})")

    # Честные предсказания для графиков: out-of-fold вероятности — каждый
    # матч предсказан моделью, которая не видела его при обучении.
    best = clone(candidate_models()[best_name])
    oof_proba = cross_val_predict(best, X, y, cv=cv,
                                  method="predict_proba")[:, 1]

    # Финальную модель учим на ВСЕХ данных — для демо и важности признаков.
    final_model = clone(best).fit(X, y)

    plot_results(final_model, X, y, oof_proba, best_name)
    return best_name, final_model


# ───── 3. Графики ─────────────────────────────────────────────

def plot_results(model, X, y, oof_proba, model_name: str) -> None:
    # --- ROC-кривая (по out-of-fold вероятностям) ---
    fpr, tpr, _ = roc_curve(y, oof_proba)
    auc = roc_auc_score(y, oof_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="случайное угадывание")
    ax.set(title="ROC-кривая: прогноз победы в карте по 10-й минуте",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    savefig(fig, "07_roc_curve.png")

    # --- Важность признаков (permutation importance) ---
    # Работает для любой модели: насколько падает ROC-AUC, если перемешать
    # значения одного признака. Больше падение → важнее признак.
    perm = permutation_importance(model, X, y, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE)
    imp = pd.Series(perm.importances_mean,
                    index=[FEATURE_RU[f] for f in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=imp.values, y=imp.index, hue=imp.index,
                palette="viridis", legend=False, ax=ax)
    ax.axvline(0, c="grey", lw=1)
    ax.set(title=f"Важность признаков ({model_name}, permutation)",
           xlabel="падение ROC-AUC при перемешивании признака", ylabel="")
    savefig(fig, "08_feature_importance.png")

    # --- Матрица ошибок (по out-of-fold прогнозам) ---
    oof_pred = (oof_proba >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, oof_pred,
        display_labels=["Dire", "Radiant"], cmap="Blues", ax=ax
    )
    ax.set_title("Матрица ошибок (out-of-fold)")
    savefig(fig, "09_confusion_matrix.png")


def plot_gold_vs_winrate(df: pd.DataFrame) -> None:
    """Наглядно: чем больше перевес по золоту на 10-й, тем выше шанс победы."""
    bins = [-1e9, -5000, -2000, 0, 2000, 5000, 1e9]
    labels = ["< -5k", "-5k..-2k", "-2k..0", "0..2k", "2k..5k", "> 5k"]
    df = df.copy()
    df["bucket"] = pd.cut(df["gold_adv"], bins=bins, labels=labels)
    wr = df.groupby("bucket", observed=True)["radiant_win"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=wr.index, y=wr.values, hue=wr.index,
                palette="rocket", legend=False, ax=ax)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set(title="Перевес Radiant по золоту на 10-й минуте → доля побед",
           xlabel="перевес по золоту (Radiant − Dire)",
           ylabel="доля побед Radiant", ylim=(0, 1))
    savefig(fig, "10_gold_vs_winrate.png")


# ───── 4. Демонстрация на одном матче ─────────────────────────

def demo_single_match(model) -> None:
    """Показывает прогноз на примере матча 8823581121, если он есть в репо."""
    example = os.path.join(BASE_DIR, "8823581121")
    if not os.path.exists(example):
        return
    with open(example) as f:
        match = json.load(f)
    feat = minute_features(match)
    if not feat:
        return

    X_one = pd.DataFrame([feat])[FEATURES]
    p_radiant = model.predict_proba(X_one)[0, 1]
    actual = "Radiant" if feat["radiant_win"] else "Dire"

    print("\n--- Пример: матч 8823581121 ---")
    print(f"Перевес Radiant на 10-й мин: нетворт={feat['gold_adv']:+d}, "
          f"опыт={feat['xp_adv']:+d}, киллы={feat['kills_adv']:+d}, "
          f"башни={feat['towers_adv']:+d}")
    print(f"Прогноз модели: победа Radiant с вероятностью {p_radiant:.1%}")
    print(f"На самом деле победил: {actual}")


# ───── main ───────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(f"Прогноз победы по состоянию на {MINUTE}-й минуте")
    print("=" * 60)

    df = build_dataset()
    plot_gold_vs_winrate(df)
    best_name, model = train_and_evaluate(df)
    demo_single_match(model)

    print(f"\nГрафики сохранены в: {os.path.relpath(FIG_DIR)}/")


if __name__ == "__main__":
    main()
