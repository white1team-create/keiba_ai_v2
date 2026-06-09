import pandas as pd
import numpy as np


class CrossFeatureBuilder:
    def __init__(self, df: pd.DataFrame):
        """
        dfには以下が含まれている前提：
        - horse_id
        - race_id
        - race_date
        - distance
        - race_class
        - track_type
        - popularity
        - odds
        - past_features（結合済み）
        """
        self.df = df.copy()

    # =========================
    # ① 相対能力スコア
    # =========================
    def build_relative_features(self):
        df = self.df

        # 人気補正スピード（オッズの逆数）
        df["log_odds"] = np.log(df["odds"].clip(lower=1.0))

        # レース内平均との差
        df["avg_popularity"] = df.groupby("race_id")["popularity"].transform("mean")
        df["popularity_gap"] = df["avg_popularity"] - df["popularity"]

        # past rank系の相対差
        if "horse_avg_rank_5r" in df.columns:
            df["avg_rank_gap"] = (
                df.groupby("race_id")["horse_avg_rank_5r"]
                .transform(lambda x: x.mean() - x)
            )

        return df

    # =========================
    # ② 距離適性スコア
    # =========================
    def build_distance_fit(self):
        df = self.df

        # 距離適性（あれば）
        if "horse_distance_win_rate" in df.columns:
            df["distance_fit_score"] = df["horse_distance_win_rate"]

        # 距離ランク補正（簡易）
        df["distance_bucket"] = pd.cut(
            df["distance"],
            bins=[0, 1400, 1800, 2200, 10000],
            labels=[1, 2, 3, 4]
        ).astype(float)

        return df

    # =========================
    # ③ 馬場適性スコア
    # =========================
    def build_track_fit(self):
        df = self.df

        if "horse_track_win_rate" in df.columns:
            df["track_fit_score"] = df["horse_track_win_rate"]
        else:
            df["track_fit_score"] = 0.5  # fallback

        return df

    # =========================
    # ④ クラス適性
    # =========================
    def build_class_fit(self):
        df = self.df

        if "horse_class_win_rate" in df.columns:
            df["class_fit_score"] = df["horse_class_win_rate"]
        else:
            df["class_fit_score"] = 0.5

        return df

    # =========================
    # ⑤ 展開適性（ペース簡易版）
    # =========================
    def build_pace_features(self):
        df = self.df

        # 人気上位＝先行想定（簡易proxy）
        df["expected_pace_position"] = df.groupby("race_id")["popularity"].rank()

        # 末脚馬っぽさ（過去成績ベース）
        if "horse_avg_rank_3r" in df.columns:
            df["late_speed_score"] = 1 / (df["horse_avg_rank_3r"] + 1)

        return df

    # =========================
    # ⑥ 総合EVスコア（超重要）
    # =========================
    def build_ev_score(self):
        df = self.df

        cols = [
            "distance_fit_score",
            "track_fit_score",
            "class_fit_score",
        ]

        # 存在するものだけ使う
        cols = [c for c in cols if c in df.columns]

        df["ev_score"] = df[cols].mean(axis=1)

        return df

    # =========================
    # ⑦ 全統合
    # =========================
    def build_all(self):
        df = self.df

        df = self.build_relative_features()
        df = self.build_distance_fit()
        df = self.build_track_fit()
        df = self.build_class_fit()
        df = self.build_pace_features()
        df = self.build_ev_score()

        return df


# =========================
# 実行例
# =========================
if __name__ == "__main__":
    df = pd.read_csv("data/processed/past_features.csv")

    builder = CrossFeatureBuilder(df)
    df_cross = builder.build_all()

    df_cross.to_csv("data/processed/cross_features.csv", index=False)

    print("cross_features created:", df_cross.shape)
