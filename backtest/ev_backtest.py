import pandas as pd


class EVBacktest:
    def __init__(self, df):
        self.df = df.copy()

    def run(self):

        df = self.df.copy()

        # =========================
        # EVフィルタ
        # =========================
        EV_THRESHOLD = 1.1
        df = df[df["ev_score"] > EV_THRESHOLD]

        # =========================
        # レース内順位（予測）
        # =========================
        df["pred_rank"] = df.groupby("race_id")["pred"].rank(ascending=False)

        # 1位だけ買う
        df = df[df["pred_rank"] == 1]

        # =========================
        # ケリー基準（資金管理）
        # =========================
        INITIAL_BANKROLL = 100000

        # 勝率（簡易モデル）
        df["p"] = 1 / (df["pred_rank"] + 1)

        # オッズ
        df["b"] = df["odds"] - 1

        # ケリー計算
        df["kelly"] = (df["b"] * df["p"] - (1 - df["p"])) / df["b"]

        # 負値カット
        df["kelly"] = df["kelly"].clip(lower=0)

        # 安定化（half kelly）
        df["kelly"] = df["kelly"] * 0.5

        # 賭け金
        df["bet"] = INITIAL_BANKROLL * df["kelly"]

        # =========================
        # 的中判定
        # =========================
        df["hit"] = (df["rank"] == 1).astype(int)

        # =========================
        # 払戻
        # =========================
        df["payout"] = df["bet"] * df["odds"] * df["hit"]

        # =========================
        # ROI計算
        # =========================
        total_bet = df["bet"].sum()
        total_return = df["payout"].sum()

        if total_bet == 0:
            roi = 0
        else:
            roi = total_return / total_bet

        print("TOTAL BET:", total_bet)
        print("TOTAL RETURN:", total_return)
        print("ROI:", roi)

        return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_dataset.csv")

    backtest = EVBacktest(df)
    result = backtest.run()

    result.to_csv("data/processed/backtest_result.csv", index=False)
