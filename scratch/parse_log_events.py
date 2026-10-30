import re

log_file = r"C:\Users\pasan\.gemini\antigravity-cli\brain\3dadb2e9-6f2c-4905-96e2-c06383217fa4\.system_generated\tasks\task-10472.log"
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if "WebSocket" in line:
            continue
        if any(w in line for w in ["Trade", "ORDER", "FILL", "EXIT", "STOP", "SL", "TP", "Leaderboard", "Champion", "PnL: -"]):
            print(line.strip())
