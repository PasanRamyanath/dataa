"""
Publishes trained model checkpoints and their exact training data to PRIVATE
Hugging Face repos — supports both the 9 canonical per-member repos AND the
5 Project-IMPJR v4.0 Multi-Horizon Tier repositories:
  - p-j-r-1-2-3/Project-IMPJR-v4-Models-tier4_macro
  - p-j-r-1-2-3/Project-IMPJR-v4-Models-tier3_meso
  - p-j-r-1-2-3/Project-IMPJR-v4-Models-tier2_tactical
  - p-j-r-1-2-3/Project-IMPJR-v4-Models-tier1_micro
  - p-j-r-1-2-3/Project-IMPJR-v4-Models-fusion
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from config import HF_MODEL_ORG, HF_MODEL_REPO_PREFIX, HF_TRAINING_DATA_REPO_ID, get_hf_token

log = logging.getLogger("common.hf_model_upload")

V4_MEMBER_ALIAS_MAP: dict[str, str] = {
    "tier4-hmm-macro": "tier4-hmm-regime",
    "tier4-patchtst-macro": "tier4-patchtst-ffd",
    "tier4-tft-macro": "tier4-tft-funding",
    "tier3-itransformer-meso": "tier3-itransformer-32",
    "tier3-nhits-meso": "tier3-nhits-multirate",
    "stage5-hierarchical-stacker": "fusion-v4-hierarchical",
}

V4_LEGACY_OOF_ALIAS_MAP: dict[str, list[str]] = {
    "tier4-hmm-regime": ["regime_states"],
    "tier4-patchtst-ffd": ["patchtst"],
    "tier4-tft-funding": ["tft"],
    "tier4-kronos-macro": ["kronos-finetune"],
    "tier3-lgbm-positioning": ["gbm-family"],
    "tier3-itransformer-32": ["itransformer"],
    "tier3-nhits-multirate": ["nhits"],
    "tier2-mamba-s6": ["ssm-challenger"],
    "tier2-crossasset-gat": ["sentiment-branch"],
    "tier2-cryptobert-news": ["sentiment-branch"],
}


def canonical_v4_member_name(member_name: str) -> str:
    return V4_MEMBER_ALIAS_MAP.get(member_name, member_name)


def v4_tier_folder_for(member_name: str) -> str | None:
    canon = canonical_v4_member_name(member_name)
    if canon.startswith("tier4-"):
        return "tier4_macro"
    if canon.startswith("tier3-"):
        return "tier3_meso"
    if canon.startswith("tier2-"):
        return "tier2_tactical"
    if canon.startswith("tier1-"):
        return "tier1_micro"
    if canon.startswith("fusion-v4") or canon.startswith("stage5-"):
        return "stage5_fusion"
    return None


def v4_tier_repo_id(member_name: str) -> str | None:
    folder = v4_tier_folder_for(member_name)
    if folder is None:
        return None
    suffix = "fusion" if folder == "stage5_fusion" else folder
    return f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-{suffix}"


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


def _upload_with_retry(
    api: HfApi,
    local_path: str | Path,
    path_in_repo: str,
    repo_id: str,
    repo_type: str,
    commit_message: str,
) -> None:
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


_NEVER_IN_MODEL_REPO = {"training_data_snapshot.parquet"}


def _expand_items(checkpoint_paths) -> list[tuple[Path, str]]:
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
            items += marker

        for local, dest in items:
            _upload_with_retry(api, local, dest, repo_id, "model", f"Upload {dest} for {member_name}")

        if metrics is not None:
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
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    repo_id = model_repo_id(member_name)
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir),
            token=get_hf_token(), repo_type="model",
        )
    except (EntryNotFoundError, RepositoryNotFoundError, Exception):
        return None
    return Path(path)


def restore_member(member_name: str, local_dir: str | Path, api: HfApi | None = None) -> bool:
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
    return token or get_hf_token() or os.environ.get("HF_TOKEN", "")


def is_member_completed_on_hf(
    member_name: str,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
    require_v4_peak: bool = False,
) -> bool:
    """Checks if a member has a completed metrics.json uploaded to HF."""
    canon = canonical_v4_member_name(member_name)
    target_repo = repo_id or v4_tier_repo_id(canon) or model_repo_id(canon)
    tok = resolve_hf_token(token)
    api = api or HfApi(token=tok)
    try:
        files = api.list_repo_files(repo_id=target_repo, repo_type="model")
        cand_files = [
            f for f in (
                f"models/{canon}/metrics.json",
                f"models/{member_name}/metrics.json",
                f"{canon}/metrics.json",
                f"{member_name}/metrics.json",
                "metrics.json",
            )
            if f in files
        ]
        if not cand_files:
            return False

        from huggingface_hub import hf_hub_download
        p_down = hf_hub_download(repo_id=target_repo, filename=cand_files[0], repo_type="model", token=tok)
        content_json = json.loads(Path(p_down).read_text(encoding="utf-8"))
        if content_json.get("status") == "failed":
            return False
        if require_v4_peak and not content_json.get("v4_peak_calibrated", False):
            return False
        return True
    except Exception:
        return False


def upload_member_checkpoint(
    member_name: str,
    output_root: str | Path,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> str:
    """Uploads checkpoint directory and OOF artifacts for member_name to its HF Hub repository."""
    canon = canonical_v4_member_name(member_name)
    output_path = Path(output_root)
    member_dir = output_path / member_name
    if not member_dir.exists() and (output_path / canon).exists():
        member_dir = output_path / canon
    elif not member_dir.exists():
        member_dir = output_path

    tier_repo = v4_tier_repo_id(canon)
    target_repo = repo_id or tier_repo or model_repo_id(canon)
    use_models_prefix = bool(repo_id or tier_repo)

    try:
        api = api or HfApi(token=resolve_hf_token(token))
        ensure_private_repo(target_repo, "model", api)

        files_to_upload = [f for f in member_dir.rglob("*") if f.is_file() and f.name != "training_data_snapshot.parquet"]
        normal_files = [f for f in files_to_upload if f.name != "metrics.json"]
        metric_files = [f for f in files_to_upload if f.name == "metrics.json"]

        for f in normal_files:
            rel = f.relative_to(member_dir).as_posix()
            dest = f"models/{canon}/{rel}" if use_models_prefix else rel
            _upload_with_retry(api, f, dest, target_repo, "model", f"Upload {canon} artifact: {rel}")

        # Upload OOF predictions from common.oof.OOF_DIR if available (including legacy aliases)
        try:
            from common.oof import OOF_DIR, oof_paths_for
            oof_candidates = list(oof_paths_for(canon))
            if member_name != canon:
                for p in oof_paths_for(member_name):
                    if p not in oof_candidates:
                        oof_candidates.append(p)
            for alias_stem in V4_LEGACY_OOF_ALIAS_MAP.get(canon, []):
                alias_p = OOF_DIR / f"{alias_stem}.parquet"
                if alias_p.exists() and alias_p not in oof_candidates:
                    oof_candidates.append(alias_p)
            for oof_p in oof_candidates:
                if oof_p.exists() and oof_p.is_file():
                    _upload_with_retry(
                        api, oof_p, f"oof/{oof_p.name}", target_repo, "model",
                        f"Upload OOF predictions for {canon}: {oof_p.name}",
                    )
        except Exception as e:
            log.debug("OOF upload skipped for %s: %s", canon, e)

        for f in metric_files:
            rel = f.relative_to(member_dir).as_posix()
            dest = f"models/{canon}/{rel}" if use_models_prefix else rel
            _upload_with_retry(api, f, dest, target_repo, "model", f"Upload {canon} completion metric: {rel}")
    except Exception as e:
        log.warning("upload_member_checkpoint failed for %s (%s): %s", canon, target_repo, e)

    return target_repo


def upload_fusion_artifacts(
    fusion_name: str = "fusion-v4-hierarchical",
    output_root: str | Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
    **kwargs,
) -> str:
    """Uploads hierarchical fusion artifacts to HF Hub."""
    actual_name = kwargs.get("fusion_stage_name") or fusion_name
    return upload_member_checkpoint(actual_name, output_root or Path("."), repo_id=repo_id, token=token, api=api)


def download_all_completed_members(
    target_dir: str | Path,
    repo_id: str | None = None,
    token: str | None = None,
    api: HfApi | None = None,
) -> list[str]:
    """Downloads all completed members and their OOF files into target_dir and common.oof.OOF_DIR."""
    from huggingface_hub import hf_hub_download
    from common.oof import OOF_DIR

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    tok = resolve_hf_token(token)
    api = api or HfApi(token=tok)
    completed: list[str] = []

    repos_to_check = [repo_id] if repo_id else [
        f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-tier4_macro",
        f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-tier3_meso",
        f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-tier2_tactical",
        f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-tier1_micro",
        f"{HF_MODEL_ORG}/{HF_MODEL_REPO_PREFIX}-fusion",
    ]

    for r_id in repos_to_check:
        if not r_id:
            continue
        try:
            files = api.list_repo_files(repo_id=r_id, repo_type="model")
            for f in files:
                if f.endswith("metrics.json"):
                    parts = f.split("/")
                    member = parts[1] if len(parts) >= 2 and parts[0] == "models" else parts[0]
                    if member not in completed:
                        completed.append(member)
                    dest_metrics = target_path / member / "metrics.json"
                    if not dest_metrics.exists():
                        try:
                            dest_metrics.parent.mkdir(parents=True, exist_ok=True)
                            cached_m = hf_hub_download(repo_id=r_id, filename=f, repo_type="model", token=tok)
                            shutil.copyfile(cached_m, dest_metrics)
                        except Exception:
                            pass
                elif f.startswith("oof/") and f.endswith(".parquet"):
                    oof_name = f[len("oof/"):]
                    dest_oof = OOF_DIR / oof_name
                    if not dest_oof.exists():
                        try:
                            cached = hf_hub_download(repo_id=r_id, filename=f, repo_type="model", token=tok)
                            shutil.copyfile(cached, dest_oof)
                        except Exception:
                            pass
        except Exception:
            pass
    return completed
