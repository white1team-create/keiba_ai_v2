import pandas as pd


class FeatureEngineering:
    def __init__(self, race_df, horse_race_df, past_df, cross_df):
        self.race_df = race_df
        self.horse_race_df = horse_race_df
        self.past_df = past_df
        self.cross_df = cross_df

    # =========================
    # レース選別（重要）
    # =========================
    def filter_races(self, df):
        race_ev = df.groupby("race_id")["ev_score"].mean()

        valid_races = race_ev[race_ev > 1.05].index

        return df[df["race_id"].isin(valid_races)]

    # =========================
    # メイン処理
    # =========================
    def build(self):
        df = self.horse_race_df.copy()

        # ① race情報結合
        df = df.merge(self.race_df, on="race_id", how="left")

        # ② past_features結合
        df = df.merge(
            self.past_df,
            on=["race_id", "horse_id"],
            how="left"
        )

        # ③ cross_features結合
        df = df.merge(
            self.cross_df,
            on=["race_id", "horse_id"],
            how="left"
        )

        # ④ レース選別（EVでフィルタ）
        df = self.filter_races(df)

        # ⑤ 欠損処理
        df = df.fillna(0)

        return df


if __name__ == "__main__":
    race_df = pd.read_csv("data/processed/race.csv")
    horse_df = pd.read_csv("data/processed/horse_race.csv")
    past_df = pd.read_csv("data/processed/past_features.csv")
    cross_df = pd.read_csv("data/processed/cross_features.csv")

    fe = FeatureEngineering(race_df, horse_df, past_df, cross_df)
    final_df = fe.build()

    final_df.to_csv("data/processed/train_dataset.csv", index=False)

    print("done:", final_df.shape)
