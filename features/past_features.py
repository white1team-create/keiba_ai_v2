import pandas as pd
import numpy as np


class PastFeatureBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["race_date"] = pd.to_datetime(self.df["race_date"])
        self.df = self.df.sort_values(["horse_id", "race_date"])

    def _rolling(self, col, window):
        return (
            self.df.groupby("horse_id")[col]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    def build_horse_performance(self):
        df = self.df

        df["rank"] = df["rank"].astype(float)

        df["horse_avg_rank_3r"] = self._rolling("rank", 3)
        df["horse_avg_rank_5r"] = self._rolling("rank", 5)

        df["is_win"] = (df["rank"] == 1).astype(int)

        df["horse_win_rate_5r"] = (
            df.groupby("horse_id")["is_win"]
            .apply(lambda x: x.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )

        return df

    def build_distance_features(self):
        df = self.df

        df["is_win"] = (df["rank"] == 1).astype(int)

        df["horse_distance_win_rate"] = (
            df.groupby(["horse_id", "distance"])["is_win"]
            .apply(lambda x: x.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )

        return df

    def build_track_features(self):
        df = self.df

        df["is_win"] = (df["rank"] == 1).astype(int)

        df["horse_track_win_rate"] = (
            df.groupby(["horse_id", "track_type"])["is_win"]
            .apply(lambda x: x.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )

        return df

    def build_class_features(self):
        df = self.df

        df["is_win"] = (df["rank"] == 1).astype(int)

        df["horse_class_win_rate"] = (
            df.groupby(["horse_id", "race_class"])["is_win"]
            .apply(lambda x: x.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )

        return df

    def build_all(self):
        df = self.df

        df = self.build_horse_performance()
        df = self.build_distance_features()
        df = self.build_track_features()
        df = self.build_class_features()

        return df
