import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


class LGBMRankerModel:
    def __init__(self):
        self.model = None

    def train(self, df, target_col="rank"):
        df = df.copy()

        # group = レース単位
        groups = df.groupby("race_id").size().values

        X = df.drop(columns=[target_col, "race_id", "horse_id"])
        y = df[target_col]

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
        }

        train_data = lgb.Dataset(X, label=y, group=groups)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500
        )

        return self

    def predict(self, df):
        X = df.drop(columns=["race_id", "horse_id"], errors="ignore")
        return self.model.predict(X)

    def save(self, path="models/lgbm_model.txt"):
        self.model.save_model(path)

    def load(self, path="models/lgbm_model.txt"):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)
