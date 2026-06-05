#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Student] RL Agent upload script — Chess
Before running:
  1. Complete training in train.py and confirm model.zip exists.
  2. Fill in STUDENT_ID below.
"""

import os

from arena_client import MLArenaClient


def main():
    # --- Fill in your details ---
    STUDENT_ID     = "YOUR_ID"          # e.g. "11223344"
    COMPETITION_ID = 11                  # Competition ID — do not change
    SLOT_INDEX     = 0                  # Slot to upload to (0 ~ rl_max_slots-1)
    SLOT_NAME      = "My Chess Agent"   # Display name shown on leaderboard
    DESCRIPTION    = ""                 # Optional: brief description of your strategy
    # ----------------------------

    client = MLArenaClient(student_id=STUDENT_ID)
    if not client.api_key:
        print(f"首次執行，正在以學號 {STUDENT_ID} 領取帳號...")
        try:
            client.enroll(student_id=STUDENT_ID)
        except RuntimeError as e:
            print("-" * 50)
            print(f"帳號領取失敗：{e}")
            print(f"請確認學號 '{STUDENT_ID}' 是否正確，及是否已加入競賽白名單。")
            print("-" * 50)
            return

    agent_file   = "agent.py"
    # model.py is sent separately; the server inlines it into agent.py so that
    # 'from model import ...' resolves correctly inside the sandbox.
    model_file   = "model.py" if os.path.exists("model.py") else None
    weights_file = "model.zip" if os.path.exists("model.zip") else None

    if not os.path.exists(agent_file):
        print("找不到 agent.py，請確認檔案存在於目前目錄。")
        return
    if weights_file is None:
        print("找不到 model.zip，將只上傳 agent.py（無模型權重）。")
        print("若需要模型權重，請先執行 train.py 再重新上傳。")

    print(f"正在上傳 Agent 至槽位 #{SLOT_INDEX}...")
    try:
        client.upload_rl_slot(
            competition_id=COMPETITION_ID,
            slot_index=SLOT_INDEX,
            agent_file=agent_file,
            model_file=model_file,
            weights_file=weights_file,
            name=SLOT_NAME,
            description=DESCRIPTION or None,
        )
    except RuntimeError as e:
        print(f"上傳失敗：{e}")
        return

    print("\n目前槽位狀態：")
    client.list_rl_slots(COMPETITION_ID)


if __name__ == "__main__":
    main()
