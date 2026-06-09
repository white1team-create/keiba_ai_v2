import pandas as pd

from pipeline.feature_engineering import FeatureEngineering
from models.lgbm_ranker import LGBMRankerModel


def main():

    # =========================
    # ① データ読み込み
    # =========================
    race_df = pd.read_csv("data/processed/race.csv")
    horse_df = pd.read_csv("data/processed/horse_race.csv")
    past_df = pd.read_csv("data/processed/past_features.csv")
    cross_df = pd.read_csv("data/processed/cross_features.csv")

    # =========================
    # ② feature統合
    # =========================
    fe = FeatureEngineering(race_df, horse_df, past_df, cross_df)
    df = fe.build()

    # =========================
    # ③ 学習
    # =========================
    model = LGBMRankerModel()
    model.train(df, target_col="rank")

    # =========================
    # ④ 予測
    # =========================
    df["pred"] = model.predict(df)

    # =========================
    # ⑤ 出力（予測順）
    # =========================
    df_sorted = df.sort_values(["race_id", "pred"], ascending=[True, False])

    df_sorted.to_csv("data/processed/predictions.csv", index=False)

    print("pipeline done:", df_sorted.shape)


if __name__ == "__main__":
    main()
