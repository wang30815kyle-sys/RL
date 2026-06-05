"""
✅ The ONLY file you need to change to swap algorithms or network architecture.
Both train.py and agent.py import from here — change once, sync everywhere.
"""

from sb3_contrib import MaskablePPO  # ✅ Change to: MaskableA2C, or wrap DQN with mask

# ═══ Algorithm choice (change freely) ════════════════════════════
ALGORITHM     = MaskablePPO   # sb3-contrib algorithm with built-in action masking
POLICY        = "MlpPolicy"   # obs = (8,8,111) flattened → (7104,); MlpPolicy fits
POLICY_KWARGS = dict(          # Custom network architecture (optional)
    net_arch=[256, 256],       # Two hidden layers; try [512, 512] for more capacity
)
SAVE_PATH     = "model"        # SB3 appends .zip; upload model.zip via run.py
# ══════════════════════════════════════════════════════════════════

# ═══ 進階：超參數調整參考 ══════════════════════════════════════════
# 以下參數可傳入 ALGORITHM(...) 呼叫（在 train.py 的 main() 中設定）：
#
#   learning_rate = 3e-4   # 預設；過大→震盪，過小→收斂慢。建議：1e-4 ~ 1e-3
#   n_steps       = 2048   # 每次更新前收集步數；越大→梯度估計越穩。建議：512 ~ 4096
#   batch_size    = 64     # 須能整除 n_steps × N_ENVS。建議：32 ~ 256
#   clip_range    = 0.2    # PPO clip；越大→更新幅度越大但越不穩定
#   ent_coef      = 0.01   # 熵正則化；越大→探索越多。若 Agent 總做同一動作可試 0.05
#
# net_arch 建議（輸入維度 7104 = 8×8×111 展平）：
#   [256, 256]                        ← 預設，適合入門
#   [512, 512]                        ← 更大容量，棋盤特徵複雜時使用
#   [512, 256, 128]                   ← 遞減式，收斂較穩
#   dict(pi=[256,256], vf=[256,256])  ← policy / value 分開設計
# ══════════════════════════════════════════════════════════════════
