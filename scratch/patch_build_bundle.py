path = 'feature engineering scripts/orchestration/kaggle_relay/build_bundle.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text_norm = text.replace('\r\n', '\n')

old_token = '''    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
        log("HF_TOKEN configured successfully")'''

new_token = '''    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ.get("KAGGLE_HF_TOKEN", "")
        log("HF_TOKEN configured successfully from bundle fallback")'''

old_env = '''    os.environ["PYTHONUNBUFFERED"] = "1"
    load_hf_token()'''

new_env = '''    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FORCE_RETRAIN"] = os.environ.get("FORCE_RETRAIN", "all")
    load_hf_token()'''

assert old_token in text_norm, 'old_token not found'
assert old_env in text_norm, 'old_env not found'

text_norm = text_norm.replace(old_token, new_token).replace(old_env, new_env)

with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(text_norm)
print("Successfully patched build_bundle.py")
