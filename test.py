import requests
import json
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt


# ───── Скачать данные ─────────────────────────────────────────

path = kagglehub.dataset_download("ektarr/dota-2-pro-matches")

# Посмотреть скаченные .csv
for file in os.listdir(path):
    print(file)


# ───── 1. Загрузить данные ─────────────────────────────────────────

teams = pd.read_csv(f"{path}/teams.csv")
tournaments = pd.read_csv(f"{path}/tournaments.csv")
players = pd.read_csv(f"{path}/players.csv")
tier1 = pd.read_csv(f"{path}/tier1_games.csv")

for names, df in [
    ("teams", teams),
    ("tournaments", tournaments),
    ("players", players),
    ("tier1", tier1),
]:
    print(f"{names} → {df.shape[0]:,} строк × {df.shape[1]} колонок")


# ───── 2. EDA ─────────────────────────────────────


def EDA(df, name):
    print(f"{name} → {df.shape[0]:,} строк × {df.shape[1]} колонок")
    print("--- Типы данных и пропуски ---")
    info_df = pd.DataFrame(
        {
            "type": df.dtypes,
            "nulls": df.isnull().sum(),
            "null_pct": (df.isnull().mean() * 100).round(2),
        }
    )
    print(info_df.sort_values("nulls", ascending=False).to_string())

    print(df.describe().round(2).to_string())
    if not df.select_dtypes(include="object").empty:
        print("\n--- Категориальные признаки (топ-3) ---")
        for col in df.select_dtypes(include="object").columns:
            print(f"{col}: {df[col].value_counts().index[:3].tolist()}")


EDA(tier1, "tier1")


"""Получение подробной информации по ID игры"""
dota_game_id = 8823581121  # из колонки dota_game_id
url = f"https://api.opendota.com/api/matches/{dota_game_id}"
data = requests.get(url).json()

with open(f"{dota_game_id}", "w") as file:
    json.dump(data, file, indent=4)
