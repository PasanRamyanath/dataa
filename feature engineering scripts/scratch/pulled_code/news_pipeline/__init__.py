"""
Full-text crypto-news pipeline: polite scraper (historical backfill + live)
-> two-tier AI scoring (FinBERT-class classifier + small instruct LLM)
-> point-in-time, multi-window news features joinable onto the 10s bars.

    python -m news_pipeline.scraper backfill --start 2024-09-01 --end 2026-09-01
    python -m news_pipeline.scraper gdelt-backfill --start 2026-07-01 --end 2026-09-01
    python -m news_pipeline.llm_scorer --once        # score everything not yet scored
    python -m news_pipeline.scraper live             # keep scraping
    python -m news_pipeline.llm_scorer               # keep scoring (event-triggered)
    python -m news_pipeline.store_sync push          # ship the scored store to the Kaggle trainer
    python -m news_pipeline.refresh                  # pull + incremental scrape + GPU scoring + push
                                                     # (what NEWS_REFRESH_ON_KAGGLE=1 runs before training)

Training consumers opt in with --with-news (models/sentiment_branch,
models/gbm_family); live predictors attach the same features automatically
when their checkpoint was trained with them.
"""
