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


# ───── Загрузить данные ─────────────────────────────────────────

teams = pd.read_csv(f"{path}/teams.csv")
tournaments = pd.read_csv(f"{path}/tournaments.csv")
players = pd.read_csv(f"{path}/players.csv")
tier1 = pd.read_csv(f"{path}/tier1_games.csv")

for names, df in [("teams", teams), ("tournaments", tournaments), ("players", players)]:
    pass


"""Получение подробной информации по ID игры"""
dota_game_id = 8823581121  # из колонки dota_game_id
url = f"https://api.opendota.com/api/matches/{dota_game_id}"
data = requests.get(url).json()

with open(f"{dota_game_id}", "w") as file:
    json.dump(data, file, indent=4)
