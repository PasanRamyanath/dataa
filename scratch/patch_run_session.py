path = 'feature engineering scripts/orchestration/kaggle_relay/run_kaggle_session.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text_norm = text.replace('\r\n', '\n')

old1 = '    force_retrain_env = os.environ.get("FORCE_RETRAIN", "")'
new1 = '    force_retrain_env = os.environ.get("FORCE_RETRAIN", "all")'

old2 = '''    # Launch Live Streaming Prediction Engine with Dual T4 GPUs on Kaggle
    log.info("=" * 70)
    log.info("STARTING LIVE PREDICTION & STREAMING ENGINE ON KAGGLE DUAL-T4 VM")
    log.info("=" * 70)
    try:
        from streaming import live_stream_glue
        live_stream_glue.main()
    except Exception:
        log.exception("Live streaming loop exited")'''

new2 = '''    # Launch Live Streaming Prediction Engine if requested
    if os.environ.get("RUN_STREAMING", "0") == "1":
        log.info("=" * 70)
        log.info("STARTING LIVE PREDICTION & STREAMING ENGINE ON KAGGLE DUAL-T4 VM")
        log.info("=" * 70)
        try:
            from streaming import live_stream_glue
            live_stream_glue.main()
        except Exception:
            log.exception("Live streaming loop exited")
    else:
        log.info("All training waves successfully completed. Finalizing session cleanly.")'''

assert old1 in text_norm, 'old1 not found'
assert old2 in text_norm, 'old2 not found'

text_norm = text_norm.replace(old1, new1).replace(old2, new2)
with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(text_norm)
print("Successfully patched run_kaggle_session.py")
