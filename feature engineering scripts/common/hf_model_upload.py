"""
Publishes trained model checkpoints and their exact training data to PRIVATE
Hugging Face repos — mirrors publish_to_hf.py's security conventions exactly
(token only from HF_TOKEN env var, HfApi().upload_file only, defensive
create_repo(exist_ok=True)) but targets per-member MODEL repos
(config.HF_MODEL_ORG/{HF_MODEL_REPO_PREFIX}-{member}, repo_type="model",
private=True) plus one shared TRAINING DATA repo
(config.HF_TRAINING_DATA_REPO_ID, repo_type="dataset", private=True),
distinct from the public-by-default daily-features repo (config.HF_REPO_ID)
that build_dataset.py/publish_to_hf.py write to.

Every ensemble member's train.py calls upload_best_checkpoint() once at the
end of training, after selecting its own best checkpoint by whatever metric
that member's validation/naive_baselines.py-comparable eval loop produced —
this module doesn't pick "best", it only uploads what the caller hands it.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from config import HF_MODEL_ORG, HF_MODEL_REPO_PREFIX, HF_TRAINING_DATA_REPO_ID, get_hf_token

log = logging.getLogger("common.hf_model_upload")


def get_api() -> HfApi:
    return HfApi(token=get_hf_token())


def model_repo_id(member_name: str) -> str:
    return f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-{member_name}"


def ensure_private_repo(repo_id: str, repo_type: str, api: HfApi | None = None) -> bool:
    try:
        api = api or get_api()
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
        return True
    except Exception as e:
        log.warning("Could not ensure private repo %s (%s): %s", repo_id, repo_type, e)
        return False


def _upload_with_retry(api: HfApi, local_path: str | Path, path_in_repo: str, repo_id: str,
                        repo_type: str, commit_message: str) -> None:
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    except Exception:
        ensure_private_repo(repo_id, repo_type, api)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
        except Exception as e:
            log.warning("Upload failed for %s to %s: %s", local_path, repo_id, e)


# Uploaded separately (upload_training_data_snapshot) — the neuralforecast
# members' `glob("*")` would otherwise push a multi-GB copy into the model repo.
_NEVER_IN_MODEL_REPO = {"training_data_snapshot.parquet"}


def _expand_items(checkpoint_paths) -> list[tuple[Path, str]]:
    """Each item is a file (uploaded under its basename), a directory
    (uploaded recursively under its own name), or an explicit
    (local_path, path_in_repo) pair."""
    out: list[tuple[Path, str]] = []
    for item in checkpoint_paths:
        local, dest = (Path(item[0]), item[1]) if isinstance(item, tuple) else (Path(item), Path(item).name)
        if local.is_dir():
            out += [(f, f"{dest}/{f.relative_to(local).as_posix()}") for f in sorted(local.rglob("*")) if f.is_file()]
        elif local.is_file() and local.name not in _NEVER_IN_MODEL_REPO:
            out.append((local, dest))
    return out


def upload_best_checkpoint(
    member_name: str,
    checkpoint_paths: list,
    metrics: dict | None = None,
    api: HfApi | None = None,
) -> str:
    """Uploads every item in `checkpoint_paths` (see _expand_items) to that
    member's private model repo, plus a metrics.json sidecar if `metrics` is
    given. Returns the repo_id uploaded to.

    metrics.json is the "member finished" marker every resume check reads
    (models.train_all, orchestration.checkpoint_manager), so it always goes
    up LAST — after the weights and, on a final upload (metrics given), the
    member's OOF predictions under oof/ that the fusion stages need. A
    session killed mid-upload therefore never leaves a repo that looks
    finished but can't be restored."""
    api = api or get_api()
    repo_id = model_repo_id(member_name)
    try:
        ensure_private_repo(repo_id, "model", api)

        items = _expand_items(checkpoint_paths)
        if metrics is not None:
            from common.oof import oof_paths_for

            items += [(p, f"oof/{p.name}") for p in oof_paths_for(member_name)]
        marker = [it for it in items if it[1] == "metrics.json"]
        items = [it for it in items if it[1] != "metrics.json"]
        if metrics is None:
            items += marker  # no sidecar coming: the listed file is the marker

        for local, dest in items:
            _upload_with_retry(api, local, dest, repo_id, "model", f"Upload {dest} for {member_name}")

        if metrics is not None:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
                tmp_path = f.name
            try:
                _upload_with_retry(api, tmp_path, "metrics.json", repo_id, "model", f"Update metrics for {member_name}")
            finally:
                os.remove(tmp_path)
    except Exception as e:
        log.warning("upload_best_checkpoint failed for %s (%s): %s", member_name, repo_id, e)

    return repo_id


def download_checkpoint_file(
    member_name: str,
    filename: str,
    local_dir: str | Path,
) -> Path | None:
    """Pulls one file (typically an in-progress epoch checkpoint) from a
    member's private model repo into local_dir — the resume-across-Kaggle-
    sessions mechanism: a fresh Kaggle VM's local disk does not carry over
    the checkpoint a prior, interrupted session wrote, but a private HF
    model repo does, since upload_best_checkpoint (or a member's own
    periodic mid-training commit) already pushes there. Returns None, not
    raises, when the repo or file doesn't exist yet — 'nothing to resume
    from' is this function's expected first-run outcome, not an error."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    repo_id = model_repo_id(member_name)
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir),
            token=get_hf_token(), repo_type="model",
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return Path(path)


def restore_member(member_name: str, local_dir: str | Path, api: HfApi | None = None) -> bool:
    """Cross-session resume: if the member's HF repo holds metrics.json (it
    finished in an earlier Kaggle session), pull the repo into `local_dir`
    (paths preserved, oof/* into common.oof.OOF_DIR) and return True, so
    models.train_all skips it and the fusion stages find its OOF file and,
    for fusion-stacking-meta-learner, its pickle. Returns False — never
    raises — when the repo is missing, unfinished, or unreachable."""
    from huggingface_hub import hf_hub_download

    from common.oof import OOF_DIR

    api = api or get_api()
    repo_id = model_repo_id(member_name)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        return False
    if "metrics.json" not in files:
        return False

    local_dir = Path(local_dir)
    token = get_hf_token()
    for name in files:
        if name.startswith(".") or name in _NEVER_IN_MODEL_REPO:
            continue
        if name.startswith("oof/"):
            target = OOF_DIR / name[len("oof/"):]
        else:
            target = local_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo_id, filename=name, repo_type="model", token=token)
        shutil.copyfile(cached, target)
    return True


def upload_training_data_snapshot(
    member_name: str,
    parquet_paths: list[str | Path],
    api: HfApi | None = None,
) -> str:
    """Uploads the EXACT parquet file(s) a member's train.py trained on to
    data/{member_name}/ in the shared private training-data repo, so a
    checkpoint in the model repo is always reproducible against the data
    snapshot that actually produced it (feature/label schema can drift over
    time — see common/feature_contract.py — so re-downloading "current"
    features later would not guarantee the same training set)."""
    api = api or get_api()
    ensure_private_repo(HF_TRAINING_DATA_REPO_ID, "dataset", api)

    for p in parquet_paths:
        p = Path(p)
        path_in_repo = f"data/{member_name}/{p.name}"
        _upload_with_retry(
            api, p, path_in_repo, HF_TRAINING_DATA_REPO_ID, "dataset",
            f"Add training data snapshot for {member_name}: {p.name}",
        )

    return HF_TRAINING_DATA_REPO_ID


def resolve_hf_token(token: str | None = None) -> str:
    """Resolves Hugging Face token from explicit arg, env var, or config."""
    return token or get_hf_token() or os.environ.get("HF_TOKEN", "")


def is_member_completed_on_hf(
    member_name: str,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> bool:
    """Checks if a member has a completed metrics.json uploaded to HF."""
    target_repo = repo_id or model_repo_id(member_name)
    api = api or HfApi(token=resolve_hf_token(token))
    try:
        files = api.list_repo_files(repo_id=target_repo, repo_type="model")
        return (
            "metrics.json" in files
            or f"models/{member_name}/metrics.json" in files
            or f"{member_name}/metrics.json" in files
        )
    except Exception:
        return False


def upload_member_checkpoint(
    member_name: str,
    output_root: str | Path,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> str:
    """Uploads checkpoint directory and artifacts for member_name to HF Hub."""
    output_path = Path(output_root)
    member_dir = output_path / member_name
    if not member_dir.exists():
        member_dir = output_path
    target_repo = repo_id or model_repo_id(member_name)
    try:
        api = api or HfApi(token=resolve_hf_token(token))
        ensure_private_repo(target_repo, "model", api)

        files_to_upload = [f for f in member_dir.rglob("*") if f.is_file()]
        # Upload metrics.json last to ensure atomic completion
        normal_files = [f for f in files_to_upload if f.name != "metrics.json"]
        metric_files = [f for f in files_to_upload if f.name == "metrics.json"]

        for f in normal_files:
            rel = f.relative_to(member_dir).as_posix()
            dest = f"models/{member_name}/{rel}" if repo_id else rel
            _upload_with_retry(api, f, dest, target_repo, "model", f"Upload {member_name} artifact: {rel}")

        for f in metric_files:
            rel = f.relative_to(member_dir).as_posix()
            dest = f"models/{member_name}/{rel}" if repo_id else rel
            _upload_with_retry(api, f, dest, target_repo, "model", f"Upload {member_name} completion metric: {rel}")
    except Exception as e:
        log.warning("upload_member_checkpoint failed for %s (%s): %s", member_name, target_repo, e)

    return target_repo


def upload_fusion_artifacts(
    fusion_name: str,
    output_root: str | Path,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> str:
    """Uploads hierarchical fusion artifacts to HF Hub."""
    return upload_member_checkpoint(fusion_name, output_root, repo_id=repo_id, token=token, api=api)


def download_all_completed_members(
    target_dir: str | Path,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> list[str]:
    """Downloads all completed members into target_dir."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    api = api or HfApi(token=resolve_hf_token(token))
    completed = []
    if repo_id:
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            for f in files:
                if f.endswith("metrics.json"):
                    parts = f.split("/")
                    member = parts[1] if len(parts) >= 2 and parts[0] == "models" else parts[0]
                    completed.append(member)
        except Exception:
            pass
    return completed
