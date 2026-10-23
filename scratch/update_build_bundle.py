path = 'feature engineering scripts/orchestration/kaggle_relay/build_bundle.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text_norm = text.replace('\r\n', '\n')

old_load = '''def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        log("HF_TOKEN loaded from Kaggle Secrets")
    except Exception:
        pass
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ.get("KAGGLE_HF_TOKEN", "")
        log("HF_TOKEN configured successfully from bundle fallback")'''

new_load = '''def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        log("HF_TOKEN loaded from Kaggle Secrets")
    except Exception:
        pass
    if not os.environ.get("HF_TOKEN"):
        _tok = "__HF_TOKEN_FALLBACK__"
        if _tok and not _tok.startswith("__"):
            os.environ["HF_TOKEN"] = _tok
            log("HF_TOKEN configured successfully from bundle fallback")'''

old_build = '''    bootstrap_code = BOOTSTRAP.replace(
        'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "orchestration.kaggle_relay.run_kaggle_session")',
        f'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "{entrypoint}")'
    )'''

new_build = '''    import config
    active_token = config.get_hf_token()
    bootstrap_code = BOOTSTRAP.replace(
        'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "orchestration.kaggle_relay.run_kaggle_session")',
        f'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "{entrypoint}")'
    ).replace('__HF_TOKEN_FALLBACK__', active_token)'''

assert old_load in text_norm, 'old_load not found'
assert old_build in text_norm, 'old_build not found'

text_norm = text_norm.replace(old_load, new_load).replace(old_build, new_build)

with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(text_norm)
print("Updated build_bundle.py")
