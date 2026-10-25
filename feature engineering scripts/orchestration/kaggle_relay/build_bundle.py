"""
Packs this repo's code into ONE self-contained Python file for Kaggle, which
only ever receives a kernel's single code_file (no repo, no packages).

The generated kaggle_session_bundle.py embeds a zip of the code (no data,
checkpoints or caches) and, when run on the Kaggle VM:
  1. unpacks it to <scratch>/impjr/feature engineering scripts, where
     <scratch> is IMPJR_ROOT, else /kaggle/temp, else /tmp — never
     /kaggle/working, which is capped at 20GB and becomes kernel output;
  2. loads HF_TOKEN (with automatic validation and fallback to the embedded
     token if Kaggle Secrets is unattached or holds an expired token);
  3. pip-installs requirements-models.txt with the preinstalled CUDA torch
     pinned (so nothing swaps it for a CPU wheel); mamba-ssm/causal-conv1d
     are attempted separately with a time limit (INSTALL_MAMBA=0 skips) —
     ssm_challenger uses its fast cuDNN GRU / chunked S6 block without them;
  4. runs orchestration.kaggle_relay.run_kaggle_session and exits with its
     code.
"""
from __future__ import annotations

import argparse
import base64
import io
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_NAME = "kaggle_session_bundle.py"
DEFAULT_OUT = REPO_ROOT / "dist" / BUNDLE_NAME

CODE_SUFFIXES = {".py", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg"}
EXCLUDED_DIRS = {"__pycache__", ".git", ".github", ".cache", "output", "dist", "checkpoints", "oof",
                 "vendor", "pretrained", "finetuned", "data", "tests", "kaggle_pulled", "kaggle_output",
                 "kaggle_push_staging", "kaggle_pulled_engine", "kaggle_out", "scratch"}
EXCLUDED_FILES = {BUNDLE_NAME, "kernel-metadata.json", "btcusdt-prediction-engine.py", "btcusdt-ensemble-train.py"}

DEFAULT_V4_FORCE_RETRAIN = (
    "tier4-hmm-regime,tier4-patchtst-ffd,tier4-tft-funding,tier4-kronos-macro,"
    "tier3-lgbm-positioning,tier3-itransformer-32,tier3-nhits-multirate,tier3-kronos-meso,"
    "tier2-lgbm-bookdepth,tier2-mamba-s6,tier2-crossasset-gat,tier2-cryptobert-news,"
    "tier1-lgbm-micro-1s10s,tier1-spot1s-leadlag,tier1-queue-hazard,fusion-v4-hierarchical"
)

BOOTSTRAP = r'''
import base64
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def log(msg):
    print(f"[bundle] {msg}", flush=True)


def scratch_root():
    for cand in (os.environ.get("IMPJR_ROOT"), "/kaggle/temp", "/tmp"):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return Path(cand) / "impjr"
    return Path.cwd() / "impjr"


def load_hf_token():
    _fallback_tok = "__HF_TOKEN_FALLBACK__"
    if _fallback_tok.startswith("__"):
        _fallback_tok = "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"

    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            sec_tok = UserSecretsClient().get_secret("HF_TOKEN")
            if sec_tok and sec_tok.strip():
                os.environ["HF_TOKEN"] = sec_tok.strip()
                log("HF_TOKEN loaded from Kaggle Secrets")
        except Exception:
            pass

    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _fallback_tok
        log("HF_TOKEN configured successfully from bundle fallback")

    # Verify token is not expired when running online on Kaggle
    if os.environ.get("HF_TOKEN") != _fallback_tok and os.environ.get("HF_TOKEN") != "test-token-offline":
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).whoami()
        except Exception as exc:
            if "401" in str(exc) or "expired" in str(exc).lower() or "unauthorized" in str(exc).lower():
                log("Detected expired/invalid HF_TOKEN in environment/secrets — switching to valid bundle fallback token")
                os.environ["HF_TOKEN"] = _fallback_tok

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def install_requirements(code):
    req_lines = (code / "requirements-models.txt").read_text(encoding="utf-8").splitlines()
    kept = [l for l in req_lines if "mamba-ssm" not in l.lower() and "causal-conv1d" not in l.lower()]
    req = code / "requirements-kaggle.txt"
    req.write_text("\n".join(kept) + "\n", encoding="utf-8")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "-r", str(req)]
    try:
        import torch
        pins = [f"torch=={torch.__version__}"]
        for extra in ("torchvision", "torchaudio"):
            try:
                pins.append(f"{extra}=={__import__(extra).__version__}")
            except Exception:
                pass
        constraints = code / "constraints-kaggle.txt"
        constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
        cmd += ["-c", str(constraints)]
        log(f"pinning preinstalled {', '.join(pins)} (CUDA available: {torch.cuda.is_available()}, "
            f"GPUs: {torch.cuda.device_count()})")
    except Exception:
        log("torch not preinstalled — installing it from requirements")
    log("pip install -r requirements-models.txt ...")
    subprocess.run(cmd, check=True)
    if os.environ.get("INSTALL_MAMBA", "0") == "1":
        log("trying mamba-ssm + causal-conv1d (optional, 20 min limit)")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
                                "causal-conv1d>=1.2.0", "mamba-ssm>=2.2"], timeout=1200)
            log("mamba-ssm installed" if r.returncode == 0 else "mamba-ssm unavailable — ssm_challenger uses its GRU fallback")
        except subprocess.TimeoutExpired:
            log("mamba-ssm build timed out — ssm_challenger uses its GRU fallback")


def main():
    import warnings
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    root = scratch_root()
    code = root / "feature engineering scripts"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as zf:
        zf.extractall(code)
    log(f"code unpacked to {code}")
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["PYTHONPATH"] = str(code) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["FORCE_RETRAIN"] = os.environ.get(
        "FORCE_RETRAIN",
        "__DEFAULT_V4_FORCE_RETRAIN__",
    )
    load_hf_token()
    if os.environ.get("SKIP_PIP_INSTALL") != "1":
        install_requirements(code)
    entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "orchestration.kaggle_relay.run_kaggle_session")
    log(f"launching Kaggle pipeline entrypoint: {entrypoint}")
    rc = subprocess.run([sys.executable, "-m"] + entrypoint.split(), cwd=str(code)).returncode
    log(f"session exited with code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
'''


def iter_code_files(root: Path = REPO_ROOT):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.suffix not in CODE_SUFFIXES or path.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS or (part.startswith("kaggle_") and part != "kaggle_relay") or part.startswith("test") for part in rel.parts[:-1]):
            continue
        yield path, rel.as_posix()


def build(out_path: Path = DEFAULT_OUT, root: Path = REPO_ROOT, entrypoint: str = "orchestration.kaggle_relay.run_kaggle_session") -> Path:
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, arcname in iter_code_files(root):
            zf.write(path, arcname)
            n += 1
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    lines = [payload[i:i + 100] for i in range(0, len(payload), 100)]
    import config
    active_token = config.get_hf_token()
    bootstrap_code = BOOTSTRAP.replace(
        'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "orchestration.kaggle_relay.run_kaggle_session")',
        f'entrypoint = os.environ.get("KAGGLE_ENTRYPOINT", "{entrypoint}")'
    ).replace('__HF_TOKEN_FALLBACK__', active_token).replace('__DEFAULT_V4_FORCE_RETRAIN__', DEFAULT_V4_FORCE_RETRAIN)
    body = ('"""Generated by orchestration/kaggle_relay/build_bundle.py — do not edit. '
            f'Self-contained training session: {n} repo files embedded below."""\n'
            "BUNDLE_B64 = (\n" + "".join(f'    "{l}"\n' for l in lines) + ")\n" + bootstrap_code)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--entrypoint", default="orchestration.kaggle_relay.run_kaggle_session")
    args = parser.parse_args()
    out = build(Path(args.out), entrypoint=args.entrypoint)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KiB) with entrypoint {args.entrypoint}")


if __name__ == "__main__":
    main()
