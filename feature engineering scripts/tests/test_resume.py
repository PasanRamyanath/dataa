import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import config
from common.cv import purged_walk_forward_splits
from common.data import filter_valid, load_days
from models.ssm_challenger.train import CHECKPOINT_DIR, train_and_eval

df = filter_valid(load_days(max_days=None))
val_size = 1 * config.BARS_PER_DAY
split = purged_walk_forward_splits(df, n_folds=1, val_size=val_size)[0]

resume_path = CHECKPOINT_DIR / "inprogress.pt"
resume_path.parent.mkdir(parents=True, exist_ok=True)
if resume_path.exists():
    resume_path.unlink()

print("=== First call: epochs=2, resume=False (simulates an interrupted 4-epoch run after epoch 1) ===")
model1, metrics1, _, _ = train_and_eval(
    split.train, split.val, lookback=50, stride=200, epochs=2, batch_size=32,
    lr=1e-3, device="cpu", resume=False, resume_path=resume_path, save_every=1,
)
assert resume_path.exists(), "inprogress.pt should exist after first (unfinished-relative-to-4) call"
import torch
ckpt = torch.load(resume_path, map_location="cpu")
assert ckpt["epoch"] == 1, f"expected checkpoint epoch=1, got {ckpt['epoch']}"
print("First call OK, checkpoint epoch =", ckpt["epoch"])

print("=== Second call: epochs=4, resume=True -> should resume from epoch 2 ===")
model2, metrics2, _, _ = train_and_eval(
    split.train, split.val, lookback=50, stride=200, epochs=4, batch_size=32,
    lr=1e-3, device="cpu", resume=True, resume_path=resume_path, save_every=1,
)
ckpt2 = torch.load(resume_path, map_location="cpu")
assert ckpt2["epoch"] == 3, f"expected final checkpoint epoch=3, got {ckpt2['epoch']}"
print("Second call OK, resumed and reached checkpoint epoch =", ckpt2["epoch"])

print("=== Third call: epochs=4 again, resume=True -> should detect already-complete and start fresh (epoch 0) ===")
model3, metrics3, _, _ = train_and_eval(
    split.train, split.val, lookback=50, stride=200, epochs=1, batch_size=32,
    lr=1e-3, device="cpu", resume=True, resume_path=resume_path, save_every=1,
)
ckpt3 = torch.load(resume_path, map_location="cpu")
assert ckpt3["epoch"] == 0, f"expected fresh-start checkpoint epoch=0, got {ckpt3['epoch']}"
print("Third call OK (correctly ignored stale completed checkpoint), checkpoint epoch =", ckpt3["epoch"])

print("=== Fourth call: mismatched fingerprint (different lookback) -> should ignore and start fresh ===")
model4, metrics4, _, _ = train_and_eval(
    split.train, split.val, lookback=60, stride=200, epochs=1, batch_size=32,
    lr=1e-3, device="cpu", resume=True, resume_path=resume_path, save_every=1,
)
ckpt4 = torch.load(resume_path, map_location="cpu")
assert ckpt4["fingerprint"]["lookback"] == 60, "fingerprint should reflect the new run's lookback"
print("Fourth call OK, fresh fingerprint =", ckpt4["fingerprint"])

print("ALL RESUME LOGIC TESTS PASSED")
