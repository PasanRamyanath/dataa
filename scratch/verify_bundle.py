with open('feature engineering scripts/kaggle_push_staging_v4/btcusdt-v4-ensemble-train.py', 'r', encoding='utf-8') as f:
    text = f.read()
print('Has FORCE_RETRAIN:', 'FORCE_RETRAIN' in text)
print('Has bundle fallback token:', 'KAGGLE_HF_TOKEN' in text)
import base64, zipfile, io
b64 = text.split('BUNDLE_B64 = (\n')[1].split(')\n')[0].replace('"', '').replace('\n', '').replace(' ', '')
zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(b64)))
print('common/data.py in zip:', 'common/data.py' in zf.namelist())
data_py = zf.read('common/data.py').decode('utf-8')
print('common/data.py has import os:', 'import os' in data_py)
cm_py = zf.read('orchestration/checkpoint_manager_v4.py').decode('utf-8')
print('checkpoint_manager_v4 has force_retrain:', 'force_retrain' in cm_py)
