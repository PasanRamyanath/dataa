# How to run the Binance Futures Prediction Ensemble

This guide walks through the whole system in order, from an empty machine to live paper-trading signals. Each step says what it does in plain English, then gives the exact commands.

The system has four parts:

1. **Data.** Download Binance/Bybit market files, turn each day into one table of 10-second bars with about 200 features and labels, and upload those daily tables to a Hugging Face (HF) dataset.
2. **Training.** On a Kaggle notebook with two T4 GPUs, train 9 models: GBM family, HMM regime, sentiment branch, SSM challenger, TFT, PatchTST, iTransformer, N-HiTS and Kronos. Then train the 3 fusion stages that combine them: stacking, meta-labeling and calibration. Every finished model is uploaded to its own private HF model repo.
3. **Validation.** Check each model against naive baselines and drop the ones that don't beat them.
4. **Live.** Load the trained models, score each new bar, simulate paper trades, and stream signals out over a Cloudflare tunnel and webhooks. No real order is ever sent to an exchange; the code has no path to do that.

> **Read "Known gaps" (section 12) before you plan a live run.** The most important one: the live loop does not yet read a live exchange feed. It scores bars that `build_dataset.py` has already written.

---

## 0. Conventions used in this guide

- **Code folder:** `F:\Downloads\Binance data\feature engineering scripts`, called the *code folder* below. Every command runs from inside it.
- **Local commands** are written for **Windows PowerShell**.
- **Kaggle commands** go in notebook cells. Lines starting with `!` are shell commands; other lines are Python.
- On Windows, set this once per terminal so log lines with special characters don't crash:
  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  ```
- Almost every script accepts `--help`, for example `python -m models.gbm_family.train --help`.

---

## 1. One-time setup

### 1.1 Accounts and secrets you need

| What | Where to get it | Used for |
|---|---|---|
| Hugging Face **write** token | huggingface.co → Settings → Access Tokens | Uploading the dataset, model checkpoints and news store |
| Kaggle API key (`kaggle.json`: username + key) | kaggle.com → Settings → API → Create New Token | Automatic relaunching of training sessions only |
| A private GitHub repo | github.com | Running the automatic relaunch every 10 minutes (optional) |
| Cloudflare account, a domain on Cloudflare, and a **named tunnel token** | Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create tunnel | A stable public URL for live signals (optional) |
| Discord / Slack webhook URLs | Your Discord channel or Slack app settings | Pushing signals to chat (optional) |
| Whale Alert API key | whale-alert.io | One optional news source (optional) |

On Kaggle, store secrets under **Add-ons → Secrets** and attach them to each notebook that needs them. Use exactly these labels:

- `HF_TOKEN`
- `CLOUDFLARE_TUNNEL_TOKEN`

### 1.2 Local Python environment (Windows)

Create the virtual environment **outside** the code folder. The Kaggle bundle builder packs every `.py` file inside the code folder, so a venv inside it would be packed too.

```powershell
cd "F:\Downloads\Binance data"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd "F:\Downloads\Binance data\feature engineering scripts"

# Data pipeline only (enough for building the dataset)
pip install -r requirements.txt

# Everything (training, fusion, validation, news scoring, live loop)
pip install -r requirements.txt -r requirements-models.txt
```

Notes:
- For a GPU on your own PC, first install a CUDA build of PyTorch from pytorch.org. The plain `torch` wheel is CPU-only.
- `mamba-ssm` only installs on Linux. On Windows, the SSM challenger automatically uses its GRU fallback.
- Kronos training needs `git` on your PATH, because it clones the Kronos repo at runtime.

### 1.3 Environment variables

Set these in PowerShell with `$env:NAME = "value"`, or in a Kaggle cell with `os.environ["NAME"] = "value"`. Only `HF_TOKEN` is always required.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(none, required)* | Your HF token |
| `HF_REPO_ID` | `p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset` | HF dataset that holds the daily feature files |
| `HF_DATA_PREFIX` | `data_v2` | Folder inside that dataset |
| `HF_MODEL_ORG` / `HF_MODEL_REPO_PREFIX` | `p-j-r-1-2-3` / `Project-IMPJR-Models` | Model repos are named `{org}/{prefix}-{member}` |
| `HF_NEWS_REPO_ID` | `{org}/Project-IMPJR-News` | Private dataset for the scored news store |
| `PROJECT_ROOT` | `F:\Downloads\Binance data` | Base folder; the other default paths hang off it |
| `BINANCE_DATA_ROOT`, `BYBIT_TRADING_ROOT`, `BYBIT_PREMIUM_INDEX_ROOT`, `BYBIT_SPOT_INDEX_ROOT` | `PROJECT_ROOT\data`, `\trading`, `\premium_index`, `\spot_index` | Where raw downloaded files live |
| `FEATURE_OUTPUT_ROOT` | `code folder\output` | Where daily feature parquets are written |
| `TRAIN_WITH_NEWS` | `auto` | `auto` uses news if a scored store exists; `1` requires it; `0` turns it off |
| `NEWS_REFRESH_ON_KAGGLE` | `0` | `1` makes each Kaggle training session scrape and score fresh news first |
| `NEWS_LLM_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | LLM used to score news |
| `NEWS_LLM_ENABLED` | `1` | `0` uses only the small tier-1 classifier |
| `NEWS_LLM_LOAD_IN_4BIT` | `0` | `1` loads a bigger (7B) scorer in 4-bit on a T4 |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | *(none)* | For the relaunch script |
| `KAGGLE_KERNEL_SLUG` | `btcusdt-ensemble-train` | Name of the Kaggle training notebook |
| `KAGGLE_SESSION_TIME_BUDGET_HOURS` | `11.5` | A training session stops cleanly after this many hours |
| `TRAIN_SKIP` | *(empty)* | Comma-separated members to skip on Kaggle, e.g. `kronos-finetune` |
| `INSTALL_MAMBA` | `1` | `0` skips the slow mamba-ssm build on Kaggle |
| `STREAM_SHARED_SECRET` | *(empty)* | Password that callers must send in the `X-Stream-Secret` header |
| `STREAM_PORT` | `8787` | Port of the local SSE server |
| `DISCORD_WEBHOOK_URL`, `SLACK_WEBHOOK_URL`, `GENERIC_WEBHOOK_URL`, `GENERIC_WEBHOOK_AUTH_HEADER` | *(empty)* | Webhook targets; an empty value means that channel is off |
| `WHALE_ALERT_API_KEY` | *(empty)* | Optional news source |

Local example:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
```

---

## 2. Step 1: Build the dataset

This step turns raw exchange files into one feature-and-label parquet per day: `features_BTCUSDT_YYYY-MM-DD.parquet`. Choose **2A** for a quick local test, or **2B** for the real multi-month backfill.

### 2A. Locally, from data already on disk

This uses the raw files under `F:\Downloads\Binance data\data`, `trading`, `premium_index` and `spot_index`.

```powershell
# Build one or more days, write them to .\output\
python build_dataset.py --start 2026-09-06 --end 2026-09-06

# Same, and also upload each day to the HF dataset
python build_dataset.py --start 2026-09-06 --end 2026-09-06 --upload
```

The two downloader scripts in the parent folder only fetch the **latest** sample of each file type. They are not a historical backfill.

```powershell
python "..\binance_btc_usd_downloader.py"
python "..\bybit_btc_usd_downloader.py"
```

### 2B. On a Kaggle CPU notebook (recommended for the full history)

`kaggle_run_pipeline.py` handles one day at a time:

1. Download exactly that day's files.
2. Build the features.
3. Upload them to HF.
4. Delete the raw files.

It first asks HF which days are already uploaded, so you can re-run it as many times as needed and it continues where it stopped. It stops itself before Kaggle's 12-hour limit.

**On your PC, build the code bundle once.** This creates `dist\kaggle_session_bundle.py`, a single file containing all the code.

```powershell
python -m orchestration.kaggle_relay.build_bundle
```

**On Kaggle:**

1. Create a new **private Dataset** named, for example, `impjr-code`, and upload `dist\kaggle_session_bundle.py` into it.
2. Create a new Notebook:
   - **Accelerator: None** (CPU, so it doesn't use any GPU quota)
   - **Internet: On**
   - **Add Input:** the `impjr-code` dataset
   - **Add-ons → Secrets:** attach `HF_TOKEN`
3. Run this cell. It unpacks the code without starting training.

```python
import os, io, sys, base64, zipfile
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
SCRATCH = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
CODE = f"{SCRATCH}/impjr/feature engineering scripts"

sys.path.insert(0, "/kaggle/input/impjr-code")
import kaggle_session_bundle as b
zipfile.ZipFile(io.BytesIO(base64.b64decode(b.BUNDLE_B64))).extractall(CODE)

# Keep raw downloads and outputs on scratch disk, never in /kaggle/working (20 GB cap)
os.environ.update({
    "PROJECT_ROOT": f"{SCRATCH}/impjr",
    "BINANCE_DATA_ROOT": f"{SCRATCH}/raw/data",
    "BYBIT_TRADING_ROOT": f"{SCRATCH}/raw/trading",
    "BYBIT_PREMIUM_INDEX_ROOT": f"{SCRATCH}/raw/premium_index",
    "BYBIT_SPOT_INDEX_ROOT": f"{SCRATCH}/raw/spot_index",
    "FEATURE_OUTPUT_ROOT": f"{SCRATCH}/output",
    "FEATURE_CACHE_ROOT": f"{SCRATCH}/cache",
})
os.chdir(CODE)
```

4. Install the dependencies and run the backfill. The plan uses about 24 months as the main training window.

```python
!pip install -q -r requirements.txt
!python kaggle_run_pipeline.py --start 2024-09-01 --end 2026-09-30
```

5. When the session ends, click **Run All** again until every day is uploaded. Already-uploaded days are skipped.

**About the 2021–2023 "regime holdout":**
- Do **not** build those years into the default `HF_DATA_PREFIX`. Every training script loads every day under that prefix, so the holdout would end up in training.
- If you want those days ready for later, set `os.environ["HF_DATA_PREFIX"] = "holdout_v2"` before running the pipeline for 2021-01-01..2023-12-31.
- Nothing scores the models on that holdout yet; see section 12.

---

## 3. Step 2 (optional): Build the news store

Two models, `gbm-family` and `sentiment-branch`, can also use news features. This step collects past crypto news articles, scores each one with a language model, and pushes **only the scored rows** to a private HF dataset. Kaggle training then pulls them from there. Skip this whole step to train without news.

Run it on your PC, or on any machine with internet. Scoring is much faster with a GPU.

```powershell
# 1. Collect past articles from the configured news sites' sitemaps
python -m news_pipeline.scraper backfill --start 2024-09-01 --end 2026-09-30

#    and/or from GDELT (add --headlines-only to skip downloading article bodies)
python -m news_pipeline.scraper gdelt-backfill --start 2024-09-01 --end 2026-09-30

#    Optional flags: --sites coindesk,decrypt   --max-articles 500

# 2. Score every new article once, then exit (--no-llm = small classifier only, no GPU needed)
python -m news_pipeline.llm_scorer --once

# 3. Push the scored store to HF so the Kaggle session can pull it
python -m news_pipeline.store_sync push
```

Alternatively, set `NEWS_REFRESH_ON_KAGGLE=1` on the training notebook (see section 4). Each training session will then top up the news store itself before training: it scrapes, scores on both T4s, and pushes. `TRAIN_WITH_NEWS=auto` (the default) uses news whenever a scored store exists.

---

## 4. Step 3: Quick sanity checks before spending GPU hours

These run on CPU and tell you early whether the features carry any signal.

```powershell
# Naive baselines (majority class / persistence) that every model must beat
python -m validation.naive_baselines

# Which feature families matter (permutation + SHAP over 3 purged time slices)
python -m validation.feature_pruning --n-folds 3
```

Add `--max-days 30` to either command for a fast trial run. `feature_pruning` writes a report only. The models do **not** read it automatically, so dropping a feature family is a manual decision (see section 12).

---

## 5. Step 4: Train the ensemble on Kaggle (dual T4)

Every member uploads its best checkpoint and its out-of-fold predictions to its own private HF repo. A new session first restores whatever is already finished, then trains only what is left. Training the whole ensemble once is roughly **95–135 GPU-hours**, so expect several 12-hour sessions spread over a few weeks of Kaggle quota.

### 5A. Manual sessions (simplest)

1. On your PC, rebuild the bundle so it contains the current code:
   ```powershell
   python -m orchestration.kaggle_relay.build_bundle
   ```
2. Upload the new `dist\kaggle_session_bundle.py` to your `impjr-code` Kaggle dataset as a new version.
3. Create a Notebook:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - **Input:** `impjr-code`
   - **Secret:** `HF_TOKEN`
4. Optionally, set options in a first cell. Child processes inherit them.
   ```python
   import os
   os.environ["TRAIN_WITH_NEWS"] = "auto"          # "0" to train without news
   os.environ["NEWS_REFRESH_ON_KAGGLE"] = "0"      # "1" = scrape and score news first
   os.environ["TRAIN_SKIP"] = ""                   # e.g. "kronos-finetune,tft"
   os.environ["INSTALL_MAMBA"] = "1"               # "0" skips the ~20 min mamba build (GRU fallback)
   ```
5. Run the training session:
   ```python
   !python /kaggle/input/impjr-code/kaggle_session_bundle.py
   ```
   The bundle unpacks the code to scratch disk and installs the requirements while keeping Kaggle's CUDA PyTorch. It then runs `models.train_all`. Training runs in this order:
   - gbm-family
   - hmm-regime
   - sentiment-branch
   - ssm-challenger
   - tft
   - patchtst
   - itransformer
   - nhits
   - kronos-finetune
   - then the fusion stages: stacking → meta-labeling → calibration
6. When the session ends, or stops itself at 11.5 hours, start it again. The next session continues where the last one stopped.
7. Check progress from your PC at any time. The command exits with code 0 once everything is done.
   ```powershell
   python -m orchestration.checkpoint_manager --check-complete
   ```

For the first session, look at the log line `Kaggle session starting: ... N GPU(s)` and confirm that it says **2 GPU(s)**.

### 5B. Automatic relaunch every 10 minutes (GitHub Actions)

The workflow `.github/workflows/kaggle_relay.yml` checks the Kaggle notebook every 10 minutes. If the last session has ended and the ensemble isn't finished, it rebuilds the bundle and pushes a new session. Once everything is trained, it does nothing.

1. Create a **private** GitHub repo whose **root is the contents of the code folder**. The `.github` folder must sit at the repo root.
   ```powershell
   cd "F:\Downloads\Binance data\feature engineering scripts"
   git init
   git add .
   git commit -m "Initial import"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Before you `git add`, consider adding a `.gitignore` that excludes `output/`, `dist/` and `**/checkpoints/`.
2. In the GitHub repo, go to **Settings → Secrets and variables → Actions** and add:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
   - `HF_TOKEN`
3. Do the **first** push yourself. The poller refuses to create a brand-new notebook unless you allow it.
   ```powershell
   pip install kaggle
   $env:KAGGLE_USERNAME = "<your kaggle username>"
   $env:KAGGLE_KEY = "<your kaggle key>"

   # See what it would do, without pushing
   python -m orchestration.kaggle_relay.poll_and_relaunch --dry-run

   # Actually create and start the notebook
   python -m orchestration.kaggle_relay.poll_and_relaunch --allow-first-push
   ```
4. Open the new notebook on kaggle.com (named after `KAGGLE_KERNEL_SLUG`):
   - Attach the `HF_TOKEN` secret.
   - Check that the accelerator is **GPU T4 x2** and Internet is **On**.
   - Re-run it once.
5. Run `kaggle kernels status <username>/btcusdt-ensemble-train` once and compare the printed status words with `ACTIVE_STATUSES` / `ENDED_STATUSES` in `orchestration/kaggle_relay/poll_and_relaunch.py`. They were written from the documentation and never checked against a live call.
6. From then on the workflow relaunches by itself. You can also trigger it by hand from the repo's **Actions** tab (**Run workflow**).

### 5C. Running single members by hand (debugging)

Inside a Kaggle notebook where the code is unpacked, as in 2B, with `HF_TOKEN` set:

```python
!pip install -q -r requirements-models.txt
!python -m models.gbm_family.train --n-folds 5 --upload            # add --with-news to use news
!python -m models.hmm_regime.train --n-states 3 --upload
!python -m models.sentiment_branch.train --upload                  # --with-news, --ablation
!python -m models.ssm_challenger.train --epochs 10 --upload --commit-to-hf   # resumes by default; --no-resume to start over
!python -m models.tft.train --max-steps 1000 --upload
!python -m models.patchtst.train --max-steps 1000 --upload
!python -m models.itransformer.train --max-steps 1000 --upload
!python -m models.nhits.train --max-steps 1000 --upload
!python -m models.kronos_finetune.train --upload --resume          # --skip-finetune = zero-shot only

# Or everything (members + fusion) in one go:
!python -m models.train_all --upload --budget-hours 11.5
```

Useful common flags:
- `--max-days N`: use only the newest N days, for a quick test.
- `--val-days N`: size of the final validation window (default 7).
- `--gpus 1`: force a single GPU.

---

## 6. Step 5: Fusion stages (when run separately)

`train_all` already runs these after the members. Run them by hand only if you retrained a member or removed one. Keep the order: meta-labeling and calibration both load the stacking model. Use the same `--val-frac` for all three.

```powershell
python -m fusion.stacking_meta_learner --val-frac 0.2 --upload
python -m fusion.meta_labeling        --val-frac 0.2 --upload
python -m fusion.calibration          --val-frac 0.2 --upload
```

They read every model's out-of-fold predictions from `fusion/oof/`. A fresh Kaggle session restores these from HF when it restores the members.

---

## 7. Step 6: Validation and kill criteria

These need the members' out-of-fold predictions in `fusion/oof/`, so run them on the machine where training ran. Alternatively, restore first (section 8).

```powershell
# Which members beat the naive baseline by >= 1% on >= 2 of 3 time slices
python -m validation.regime_holdout_eval --relative-threshold 0.01

# Does a proposed branch beat what it would replace?
python -m validation.ablation_test --variant sentiment-branch --baseline gbm-family
python -m validation.ablation_test --variant multi_asset_btc --baseline ssm-challenger
```

The report is written to `validation/checkpoints/regime_holdout_eval/metrics.json` and lists `keep` and `exclude`.

**The live system does not read the `exclude` list automatically.** To actually drop a failing member:
1. Delete its file `fusion/oof/<member>.parquet` and its `models/<member>/checkpoints/` folder.
2. Re-run the three fusion commands from section 6.

---

## 8. Step 7: Get trained models onto the machine that runs live inference

The live loop only loads checkpoints from local folders, so download them from HF first. This one command restores every finished member and fusion stage. Run it from the code folder with `HF_TOKEN` set.

```powershell
python -c "from models.train_all import MEMBERS, FUSION_STAGES; from common.hf_model_upload import restore_member; [print(n, restore_member(n, d)) for _, n, d, *_ in MEMBERS]; [print(n, restore_member('fusion-' + n, d)) for _, n, d in FUSION_STAGES]"
```

Each line prints `True` if that piece was restored.

Kronos has one extra requirement: the vendored Kronos repo must exist at `models/kronos_finetune/vendor/Kronos`. Run any Kronos training command once on that machine (with `--skip-finetune --max-days 1` it is quick), or `git clone https://github.com/shiyu-coder/Kronos models/kronos_finetune/vendor/Kronos`. Without it, Kronos is skipped and the rest still works.

---

## 9. Step 8: Run the live system

**What "live" means today.** The loop checks every second for new rows in the newest `features_BTCUSDT_*.parquet` under `FEATURE_OUTPUT_ROOT`.
- When a new bar appears, every loaded model scores it, the fusion stages combine the scores, and the paper-trading engine may open or close a simulated position.
- To feed it, keep running `build_dataset.py` for the newest day (section 2A).

Binance Vision publishes daily files about a day late, so today this is a near-live replay, not a real-time feed (see section 12).

Every component is optional and degrades gracefully. A missing model is skipped with a log line instead of crashing the loop.

### 9.1 Smoke tests (no trained models needed)

```powershell
python -m inference.paper_trading_engine --demo                # simulated self-check of sizing / circuit breakers
python -m inference.live_loop --once --no-trading              # one tick, inference only
python -m streaming.cloudflared_manager --status               # is a tunnel token / binary present?
python -m streaming.webhook_dispatcher --demo                  # offline preview of webhook payloads
python -m resource_maximization.online_adapters --demo
python -m resource_maximization.drift_gate --demo
python -m resource_maximization.mc_dropout_uncertainty --demo
```

`webhook_dispatcher --demo` writes `streaming/data/sequence_state.json`. Delete that file afterwards so that real runs start numbering from 1.

### 9.2 Basic live loop with paper trading

```powershell
python -m inference.live_loop                    # run forever (Ctrl+C to stop)
python -m inference.live_loop --max-cycles 60    # stop after 60 ticks
python -m inference.live_loop --no-trading       # predictions only
```

Output locations:
- Predictions: `inference/data/tick_log/`
- Simulated trades and equity: `inference/data/paper_trading/`

### 9.3 Live news (only if gbm-family / sentiment-branch were trained with news)

Run these in two extra terminals next to the live loop. The models then see fresh news features.

```powershell
python -m news_pipeline.scraper live          # polls the news sites every 30 s
python -m news_pipeline.llm_scorer            # scores new articles as they arrive (add --no-llm for CPU-only)
```

`inference.news_feed` (RSS / GDELT / Whale Alert) and `inference.sentiment_scorer` are an older, separate news path. Nothing in the live loop reads their output, so you don't need to run them.

### 9.4 With resource maximization (§13)

This adds drift-gated GBM adaptation, which always starts from the last full retrain, plus MC-dropout uncertainty on the SSM member.

```powershell
python -m resource_maximization.live_glue                     # run forever
python -m resource_maximization.live_glue --no-adaptation     # uncertainty only
python -m resource_maximization.live_glue --once --no-trading # one-tick wiring test
```

Optional multi-asset model. It needs ETH/SOL/BNB feature data, which the pipeline can't build yet (see section 12):

```powershell
python -m resource_maximization.multi_asset_train --epochs 5 --upload
```

### 9.5 With streaming (SSE endpoint, Cloudflare tunnel, webhooks)

Set the secrets first:

```powershell
$env:STREAM_SHARED_SECRET = "pick-a-long-random-string"
$env:CLOUDFLARE_TUNNEL_TOKEN = "<named tunnel token>"     # on Kaggle use the Kaggle secret instead
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
```

In the Cloudflare dashboard, point the tunnel's public hostname at `http://localhost:8787`, or whatever `STREAM_PORT` is set to.

```powershell
python -m streaming.live_stream_glue                    # live loop + SSE server + tunnel + webhooks
python -m streaming.live_stream_glue --no-tunnel        # local only (http://localhost:8787)
python -m streaming.live_stream_glue --no-webhooks
python -m streaming.live_stream_glue --once --no-trading
```

Callers must send the shared secret:

```powershell
curl.exe -H "X-Stream-Secret: <secret>" https://<your-hostname>/latest
curl.exe -N -H "X-Stream-Secret: <secret>" https://<your-hostname>/stream
```

On Kaggle, `cloudflared_manager` downloads the official linux-amd64 `cloudflared` binary by itself. On Windows, install `cloudflared` yourself and set `CLOUDFLARED_BINARY_PATH` to it.

### 9.6 Everything together (resource maximization + streaming)

There is no single command for this yet. Save the following as `run_all_live.py` in the code folder and run `python run_all_live.py`:

```python
from resource_maximization import live_glue
from streaming.live_stream_glue import build_combined_on_tick, start_background_services
from streaming.sequencing import Sequencer
from streaming.sse_server import LiveCache
from streaming.webhook_dispatcher import WebhookDispatcher

cache = LiveCache()
dispatcher = WebhookDispatcher(sequencer=Sequencer())
start_background_services(cache, dispatcher, enable_tunnel=True)
live_glue.run_forever(external_on_tick=build_combined_on_tick(dispatcher, cache))
```

### 9.7 Running live on a Kaggle GPU notebook

1. Create a GPU T4 x2 notebook with Internet on and the `HF_TOKEN` and `CLOUDFLARE_TUNNEL_TOKEN` secrets attached.
2. Unpack the code exactly as in section 2B, step 3.
3. Run `!pip install -q -r requirements-models.txt`.
4. Restore the models (section 8).
5. Build the newest day's features (`!python build_dataset.py --start <yesterday> --end <yesterday>`).
6. Run `!python -m streaming.live_stream_glue`, or the `run_all_live.py` script from 9.6.

There is no automatic relaunch for live sessions; the relay in 5B only restarts training.

---

## 10. Retraining routine (ongoing)

1. Extend the dataset with new days by re-running the CPU notebook (2B) with a later `--end`.
2. Delete the finished-model markers so members retrain. The resume logic skips any member whose HF repo has a `metrics.json`. Either:
   - delete that file in each model repo on huggingface.co, or
   - run training with `--no-resume --no-restore-from-hf` through `models.train_all` by hand.
3. Run training sessions (5A or 5B) until `checkpoint_manager --check-complete` passes.
4. Re-run validation (section 7), drop any members that fail, re-run fusion (section 6), and restore on the live machine (section 8).

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `HF_TOKEN environment variable is not set` | Set `$env:HF_TOKEN` locally, or attach the `HF_TOKEN` secret on Kaggle (in a CPU notebook you also have to copy it into `os.environ` yourself, as in 2B). |
| `UnicodeEncodeError` printing logs on Windows | `$env:PYTHONIOENCODING = "utf-8"` |
| `No feature parquet files found` | Build the dataset first (section 2), or make sure `HF_REPO_ID` / `HF_DATA_PREFIX` point to where you uploaded it. |
| `no OOF predictions found under fusion/oof/` | At least one member must finish training before fusion. On a new machine, restore first (section 8). |
| Training session shows `1 GPU(s)` | Set the notebook accelerator to **GPU T4 x2**. Members also retry on a single GPU automatically if dual-GPU fails. |
| mamba-ssm build is slow or fails | Harmless: the SSM member uses its GRU fallback. Set `INSTALL_MAMBA=0` to skip the build. |
| A member keeps failing every session | After 3 attempts (`MAX_MEMBER_ATTEMPTS`) it is given up, and fusion runs without it. Or skip it with `TRAIN_SKIP=<name>`. |
| Kaggle `/kaggle/working` getting full | Keep all raw data and outputs under `/kaggle/temp` (or `/tmp`) as in 2B. Only `/kaggle/working` is capped at 20 GB. |
| Live loop logs "zero members vote" | No checkpoints were found locally. Run the restore command in section 8. |
| Kronos "vendored repo not present" | Clone it as described in section 8. |
| Relay never relaunches | Check that the GitHub secrets exist, that the first push was done with `--allow-first-push`, and that the Kaggle status words match (5B, step 5). |
| Webhook sequence numbers start mid-way | Delete `streaming/data/sequence_state.json` if it came from a demo run. |

---

## 12. Known gaps (what is not implemented yet)

These are the places where the code does **not** yet do what the master plan describes:

1. **No live exchange feed.** The plan (§4.5, §12.1) pulls bars from Binance REST/WebSocket every second. The live loop instead reads parquet files from `build_dataset.py`, which is built from Binance Vision daily archives published about a day late. The missing piece is a WebSocket → 10s bar → incremental-features builder.
2. **No automatic relaunch for live inference sessions.** The Kaggle relay (§11.2) only restarts *training*. Live sessions are started by hand (9.7). Rotating trade logs to a versioned Kaggle Dataset (§12.3) is also missing.
3. **The 2021–2023 regime holdout isn't evaluated.** No script scores trained members on a separate holdout dataset. Also, `regime_holdout_eval` only sees the members' recent out-of-fold window, so it falls back to three recent time slices and says so in its log.
4. **Kill criteria are reported, not enforced.** Excluded members still feed fusion until you remove them by hand (section 7).
5. **Feature pruning is report-only.** No member reads the pruning result.
6. **Multi-asset data is missing.** `build_dataset.py` is hard-wired to BTCUSDT, so `multi_asset_train` has no ETH/SOL/BNB data to train on.
7. **Single horizon only.** Models train on the 10-second-bar target only (triple-barrier with a 1-hour max hold). The separate 1s / 1-min / 5-min horizon models from §6.2 are not built; 1m/5m data appear only as context columns.
8. **Calibration is isotonic, not conformal (§8.3).** MC-dropout spread is computed live but is not fed into the calibrator.
9. **Regime gating is approximate.** It works through regime one-hot inputs to the stacker, not by multiplying member weights (§8.1). This is functionally similar, but it is not the plan's exact mechanism.
10. **No session coverage report (§12.4).** The "active X% of the period" figure is not computed or logged.
11. **The TPU training path (§11.4, optional) doesn't exist.**
12. **Older history lacks some sources.** Some Binance Vision data types, such as book depth and metrics, don't go back as far as 2021. Very old days will have those feature columns empty.
