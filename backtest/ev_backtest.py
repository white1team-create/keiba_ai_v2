# =========================
# EVフィルタ（ここが核心）
# =========================

EV_THRESHOLD = 1.1

filtered_df = df[df["ev_score"] > EV_THRESHOLD]
