import sys

fpath = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(fpath, "r", encoding="utf-8") as f:
    content = f.read()

# Let's check line endings
print("File length:", len(content))
print("CRLF in file:", "\r\n" in content)
