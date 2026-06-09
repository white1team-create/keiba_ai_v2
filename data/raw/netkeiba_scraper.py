import requests
from bs4 import BeautifulSoup
import pandas as pd

# =========================
# 1レース取得（完全安定版）
# =========================
def fetch_race(race_url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    res = requests.get(race_url, headers=headers)

    # ★ encodingを一切いじらない
    soup = BeautifulSoup(res.content, "html.parser")

    rows = soup.select("table.race_table_01 tr")

    data = []

    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 10:
            continue

        horse = {
            "rank": cols[0].get_text(strip=True),
            "horse_name": cols[3].get_text(strip=True),
            "odds": cols[9].get_text(strip=True),
        }

        data.append(horse)

    return pd.DataFrame(data)


# =========================
# 実行テスト
# =========================
if __name__ == "__main__":
    race_url = "https://db.netkeiba.com/race/202401010101/"

    df = fetch_race(race_url)

    print(df.head())

    df.to_csv("race_sample.csv", index=False, encoding="utf-8-sig")
    print("saved: race_sample.csv")
