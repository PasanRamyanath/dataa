import base64
import io
import zipfile
import pathlib

content = pathlib.Path("feature engineering scripts/kaggle_push_staging/btcusdt-prediction-engine.py").read_text(encoding="utf-8")
b64_part = content.split("BUNDLE_B64 = (\n")[1].split(")\n")[0]
b64_clean = "".join(line.strip().strip('"') for line in b64_part.splitlines())
zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(b64_clean)))
files = zf.namelist()
print("Total files in bundle:", len(files))
for key in ["live_loop", "dual_gpu", "config", "binance_ws"]:
    found = [f for f in files if key in f]
    print(f"{key}: {found}")
