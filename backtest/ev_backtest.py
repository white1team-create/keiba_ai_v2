# =========================
# EVフィルタ（ここが核心）
# =========================

EV_THRESHOLD = 1.1

filtered_df = df[df["ev_score"] > EV_THRESHOLD]

# =========================
# ベットロジック
# =========================

BET_AMOUNT = 100
filtered_df = filtered_df.copy()
filtered_df["bet"] = BET_AMOUNT

# =========================
# ROI計算
# =========================

filtered_df["hit"] = (filtered_df["rank"] == 1).astype(int)
filtered_df["payout"] = filtered_df["bet"] * filtered_df["odds"] * filtered_df["hit"]

total_bet = filtered_df["bet"].sum()
total_return = filtered_df["payout"].sum()

roi = total_return / total_bet

print("ROI:", roi)
