"""
collect_matches.py
==================
Скачивает детальную статистику матчей из OpenDota API в папку matches/.

Зачем это нужно:
  В tier1_games.csv лежит только результат серии и ID матчей. Чтобы
  предсказывать победу по состоянию игры на 10-й минуте, нужны подробные
  данные по каждой карте (золото/опыт/добивания по минутам) — их отдаёт
  OpenDota по адресу https://api.opendota.com/api/matches/{id}.

Как работает:
  1. Берём ID карт из tier1_games.csv (только has_game_data=1 и есть dota_game_id)
  2. Берём случайную, но ВОСПРОИЗВОДИМУЮ выборку N штук (random_state)
  3. Для каждого ID качаем JSON и сохраняем в matches/{id}.json
  4. Если файл уже скачан — пропускаем (скрипт можно прерывать и запускать заново)
  5. Между запросами делаем паузу, чтобы не упереться в лимит OpenDota (~60 запросов/мин)

Запуск:
    python collect_matches.py          # 400 матчей (по умолчанию)
    python collect_matches.py 1000     # столько матчей попробовать скачать
"""

from __future__ import annotations

import json
import os
import sys
import time

import kagglehub
import pandas as pd
import requests

# ───── Настройки ──────────────────────────────────────────────
N_DEFAULT = 400        # сколько матчей качаем, если число не передали аргументом
SLEEP = 1.1            # пауза между запросами (сек): держим < 60 запросов/мин
RANDOM_STATE = 42      # фикс выборки → при каждом запуске один и тот же набор
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matches")
os.makedirs(OUT_DIR, exist_ok=True)


def get_match_ids(n: int) -> list[int]:
    """Возвращает список ID матчей из tier1_games.csv для скачивания."""
    path = kagglehub.dataset_download("ektarr/dota-2-pro-matches")
    df = pd.read_csv(os.path.join(path, "tier1_games.csv"))

    # только карты, у которых есть детальная статистика и заполнен dota_game_id
    ok = df[(df["has_game_data"] == 1) & df["dota_game_id"].notna()]
    ids = ok["dota_game_id"].astype("int64").drop_duplicates()

    # случайная, но воспроизводимая выборка (один и тот же seed → тот же набор)
    ids = ids.sample(n=min(n, len(ids)), random_state=RANDOM_STATE)
    return ids.tolist()


def fetch_match(match_id: int) -> dict | None:
    """Качает один матч из OpenDota. Возвращает dict или None при ошибке."""
    url = f"https://api.opendota.com/api/matches/{match_id}"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  ! {match_id}: ошибка сети ({e})")
        return None

    # 429 = слишком много запросов → подождём подольше и попробуем ещё раз
    if r.status_code == 429:
        print("  ... лимит API, ждём 30 сек")
        time.sleep(30)
        r = requests.get(url, timeout=30)

    if r.status_code != 200:
        print(f"  ! {match_id}: HTTP {r.status_code}")
        return None
    return r.json()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    ids = get_match_ids(n)
    print(f"К скачиванию: {len(ids)} матчей → {os.path.relpath(OUT_DIR)}/")

    saved = skipped = failed = 0
    for i, mid in enumerate(ids, 1):
        out = os.path.join(OUT_DIR, f"{mid}.json")
        if os.path.exists(out):          # уже скачан раньше — не тратим запрос
            skipped += 1
            continue

        data = fetch_match(mid)
        if data is None:
            failed += 1
            continue

        with open(out, "w") as f:
            json.dump(data, f)
        saved += 1

        if i % 25 == 0:
            print(f"  [{i}/{len(ids)}] скачано={saved} "
                  f"уже было={skipped} ошибок={failed}")
        time.sleep(SLEEP)

    print(f"\nГотово. Скачано={saved}, уже было={skipped}, ошибок={failed}")
    print(f"Всего файлов в папке: {len(os.listdir(OUT_DIR))}")


if __name__ == "__main__":
    main()
