import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_ssm_dedicated")

def main():
    import config
    from common.data import ensure_training_data_ready
    from models.ssm_challenger import train as ssm_train

    log.info("Ensuring training data is ready...")
    ensure_training_data_ready(symbol=config.SYMBOL_UM, min_days=42)

    log.info("Starting dedicated SSM Challenger training on GPU (Option A: 4-Layer cuDNN GRU)...")
    metrics = ssm_train.run(
        val_days=config.VAL_DAYS,
        max_days=180,
        lookback=config.LOOKBACK_PERIODS,
        stride=10,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        upload=True,
        commit_to_hf=True,
    )
    log.info("SSM Challenger dedicated training finished successfully: %s", metrics)

if __name__ == "__main__":
    main()
