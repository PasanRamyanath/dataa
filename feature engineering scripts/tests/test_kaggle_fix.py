import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["HF_TOKEN"] = "test-token-offline"

import numpy as np

import config
import common.hf_model_upload as hmu
import common.oof as oof
import orchestration.checkpoint_manager as cm
import models.train_all as ta

TMP = Path(tempfile.mkdtemp(prefix="kfix_"))


class FakeApi:
    """In-memory HF: {(repo_id, repo_type): {path: bytes}}."""
    def __init__(self):
        self.repos = {}
        self.log = []

    def create_repo(self, repo_id, repo_type, exist_ok, private):
        assert private
        self.repos.setdefault((repo_id, repo_type), {})

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        if (repo_id, repo_type) not in self.repos:
            raise RuntimeError("404")
        self.repos[(repo_id, repo_type)][path_in_repo] = data
        self.log.append((repo_id, path_in_repo))

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, allow_patterns, commit_message):
        for f in Path(folder_path).glob(allow_patterns[0]):
            self.upload_file(str(f), f"{path_in_repo}/{f.name}", repo_id, repo_type, commit_message)

    def list_repo_files(self, repo_id, repo_type):
        if (repo_id, repo_type) not in self.repos:
            raise RuntimeError("RepositoryNotFound")
        return list(self.repos[(repo_id, repo_type)])


API = FakeApi()


def fake_download(repo_id, filename, repo_type=None, token=None, **kw):
    data = API.repos.get((repo_id, repo_type or "model"), {}).get(filename)
    if data is None:
        raise FileNotFoundError(filename)
    out = TMP / "hfcache" / repo_id.replace("/", "__") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out)


import huggingface_hub
orig_dl = huggingface_hub.hf_hub_download
huggingface_hub.hf_hub_download = fake_download
orig_get_api = hmu.get_api
hmu.get_api = lambda: API
orig_oof_dir = oof.OOF_DIR
oof.OOF_DIR = TMP / "oof"
saved_members, saved_fusion = ta.MEMBERS, ta.FUSION_STAGES
orig_run = ta._run
orig_cuda = ta._cuda_count
try:
    # ---- 1. upload ordering, dir recursion, tuple paths, snapshot skip, OOF
    ck = TMP / "ck_gbm"; ck.mkdir()
    (ck / "direction_classifier.txt").write_text("m")
    (ck / "training_data_snapshot.parquet").write_text("big")
    (ck / "metrics.json").write_text("{}")
    sub = ck / "sub"; sub.mkdir(); (sub / "a.bin").write_text("x")
    oof.save_oof("gbm-family", np.arange(3), np.full((3, 3), 1 / 3))
    tok = TMP / "tok"; tok.mkdir(); (tok / "config.json").write_text("tok")
    bm = TMP / "bm"; bm.mkdir(); (bm / "config.json").write_text("bm")
    hmu.upload_best_checkpoint("gbm-family", list(ck.glob("*")) + [(tok, "tokenizer"), (bm, "basemodel")],
                               metrics={"acc": 1}, api=API)
    repo = hmu.model_repo_id("gbm-family")
    files = API.repos[(repo, "model")]
    assert "training_data_snapshot.parquet" not in files
    assert files["tokenizer/config.json"] == b"tok" and files["basemodel/config.json"] == b"bm"
    assert "sub/a.bin" in files and "oof/gbm-family.parquet" in files
    order = [p for r, p in API.log if r == repo]
    assert order[-1] == "metrics.json" and order.count("metrics.json") == 1, order
    assert json.loads(files["metrics.json"]) == {"acc": 1}
    # mid-training upload (no metrics): no OOF, listed file still uploaded
    hmu.upload_best_checkpoint("ssm-challenger", [ck / "direction_classifier.txt"], api=API)
    assert list(API.repos[(hmu.model_repo_id("ssm-challenger"), "model")]) == ["direction_classifier.txt"]
    # hmm OOF name mapping
    oof.save_regime_states(np.arange(3), np.zeros(3))
    assert oof.oof_paths_for("hmm-regime")[0].name == "regime_states.parquet"
    print("1 upload ok")

    # ---- 2. restore_member
    shutil.rmtree(oof.OOF_DIR)
    dest = TMP / "restored"
    assert hmu.restore_member("gbm-family", dest, api=API) is True
    assert (dest / "metrics.json").exists() and (dest / "tokenizer" / "config.json").read_text() == "tok"
    assert (oof.OOF_DIR / "gbm-family.parquet").exists()
    assert hmu.restore_member("ssm-challenger", TMP / "r2", api=API) is False  # no metrics.json
    assert hmu.restore_member("tft", TMP / "r3", api=API) is False              # no repo
    print("2 restore ok")

    # ---- 3. failure ledger
    assert cm.load_failures() == {}
    assert cm.record_failure("kronos-finetune", "timeout") == 1
    assert cm.record_failure("kronos-finetune", "failed") == 2
    f = cm.load_failures()
    assert f["kronos-finetune"] == {"count": 2, "last_reason": "failed"} and not cm.given_up("kronos-finetune", f)
    cm.record_failure("kronos-finetune", "failed")
    assert cm.given_up("kronos-finetune", cm.load_failures())
    done, status = cm.ensemble_complete(API)
    assert status["gbm-family"] and status["kronos-finetune"] and not status["tft"] and not done
    print("3 ledger ok")

    # ---- 4. train_all orchestration (subprocess mocked)
    ta.MEMBERS = [
        ("m.gbm", "gbm-family", TMP / "c_gbm", False, []),
        ("m.sent", "sentiment-branch", TMP / "c_sent", False, []),
        ("m.ssm", "ssm-challenger", TMP / "c_ssm", True, []),
        ("m.tft", "tft", TMP / "c_tft", True, []),
        ("m.kronos", "kronos-finetune", TMP / "c_kronos", True, []),
    ]
    ta.FUSION_STAGES = [("f.stack", "stacking-meta-learner", TMP / "c_stack"),
                        ("f.cal", "calibration", TMP / "c_cal")]
    calls = []
    plan = {}

    def fake_run(module, args, timeout=None):
        calls.append((module, list(args), timeout))
        outs = plan.get(module, ["ok"])
        return outs.pop(0) if len(outs) > 1 else outs[0]

    ta._run = fake_run
    ta._cuda_count = lambda: 2

    class B:
        safety_margin_seconds = 900
        def __init__(self, rem): self.rem = rem
        def remaining_seconds(self): return self.rem
        def should_stop(self): return self.rem <= self.safety_margin_seconds

    plan.update({"m.tft": ["failed", "ok"], "m.ssm": ["timeout"]})
    res = ta.run(7, None, True, [], False, gpus=None, budget=B(40000), with_news=True)
    assert res["gbm-family"] == "resumed-skip"                     # restored from fake HF
    assert res["kronos-finetune"] == "given-up"                     # 3 failures
    assert res["tft"] == "ok" and res["ssm-challenger"] == "timeout"
    tft_calls = [c for c in calls if c[0] == "m.tft"]
    assert len(tft_calls) == 2 and "--gpus" not in tft_calls[0][1] and tft_calls[1][1][-2:] == ["--gpus", "1"]
    ssm_args = [c for c in calls if c[0] == "m.ssm"][0][1]
    assert "--commit-to-hf" in ssm_args and "--upload" in ssm_args and "--with-news" not in ssm_args
    assert "--with-news" in [c for c in calls if c[0] == "m.sent"][0][1]
    assert all(abs(c[2] - (40000 - 900)) < 1 for c in calls)
    assert "stacking-meta-learner" not in res                        # deferred: ssm unfinished
    assert cm.load_failures()["ssm-challenger"]["count"] == 1
    print("4a partial session ok:", res)

    # next session: ssm ok now; everything else restorable only if uploaded — simulate local metrics
    for d in ("c_sent", "c_tft"):
        (TMP / d).mkdir(exist_ok=True); (TMP / d / "metrics.json").write_text("{}")
    calls.clear(); plan.clear()
    res = ta.run(7, None, True, [], False, gpus=2, budget=B(40000))
    assert res["ssm-challenger"] == "ok" and res["stacking-meta-learner"] == "ok" and res["calibration"] == "ok"
    assert [c for c in calls if c[0] == "m.ssm"][0][1][-3:] == ["--gpus", "2", "--commit-to-hf"]
    fusion_call = [c for c in calls if c[0] == "f.stack"][0][1]
    assert fusion_call == ["--val-frac", "0.2", "--upload"]
    print("4b completing session ok:", res)

    # budget exhausted before a member -> stop, no fusion
    shutil.rmtree(TMP / "c_tft")
    for m in ("tft",):
        API.repos.pop((hmu.model_repo_id(m), "model"), None)
    calls.clear()
    res = ta.run(7, None, True, [], False, budget=B(100))
    assert "tft" not in res and not any(c[0].startswith("f.") for c in calls)
    # local run without upload: no HF, no ledger
    calls.clear()
    res = ta.run(7, None, False, ["kronos-finetune"], True)
    assert res["tft"] == "ok" and "--upload" not in calls[0][1]
    print("4c budget/local ok")

    # ---- 5. real _run: ok, failed, timeout (kills child)
    ta._run = orig_run
    mod_dir = TMP / "mods"; mod_dir.mkdir()
    (mod_dir / "kf_ok.py").write_text("pass\n")
    (mod_dir / "kf_bad.py").write_text("raise SystemExit(3)\n")
    (mod_dir / "kf_slow.py").write_text("import time\ntime.sleep(60)\n")
    os.environ["PYTHONPATH"] = str(mod_dir)
    assert ta._run("kf_ok", []) == "ok" and ta._run("kf_bad", []) == "failed"
    import time as _t
    t0 = _t.time()
    assert ta._run("kf_slow", [], timeout=2) == "timeout" and _t.time() - t0 < 20
    del os.environ["PYTHONPATH"]
    print("5 subprocess ok")

    # ---- 6. news store sync
    import news_pipeline.store_sync as ss
    orig_root = config.NEWS_ARTICLES_ROOT
    config.NEWS_ARTICLES_ROOT = TMP / "news_src"
    (TMP / "news_src" / "scored").mkdir(parents=True)
    (TMP / "news_src" / "scored" / "scored_2026-09-01.jsonl").write_text('{"a":1}\n')
    assert ss.push(API) == 1
    config.NEWS_ARTICLES_ROOT = TMP / "news_dst"
    assert ss.pull(API) == 1 and (TMP / "news_dst" / "scored" / "scored_2026-09-01.jsonl").exists()
    orig_repo = config.HF_NEWS_REPO_ID
    config.HF_NEWS_REPO_ID = "nobody/missing"
    assert ss.pull(API) == 0
    config.HF_NEWS_REPO_ID = orig_repo
    # run_kaggle_session news decision
    import orchestration.kaggle_relay.run_kaggle_session as rks
    config.TRAIN_WITH_NEWS = "auto"; assert rks.decide_with_news() is True
    config.TRAIN_WITH_NEWS = "0"; assert rks.decide_with_news() is False
    config.HF_NEWS_REPO_ID = "nobody/missing"; config.TRAIN_WITH_NEWS = "1"
    try:
        rks.decide_with_news(); raise AssertionError("should raise")
    except RuntimeError:
        pass
    config.HF_NEWS_REPO_ID = orig_repo; config.TRAIN_WITH_NEWS = "auto"; config.NEWS_ARTICLES_ROOT = orig_root
    print("6 news sync ok")

    # ---- 7. bundle: contents, unpack + imports, bootstrap install/run commands
    import orchestration.kaggle_relay.build_bundle as bb
    out = bb.build(TMP / "stage" / bb.BUNDLE_NAME)
    size_kib = out.stat().st_size / 1024
    spec = importlib.util.spec_from_file_location("bundle_mod", out)
    bundle = importlib.util.module_from_spec(spec); spec.loader.exec_module(bundle)
    import base64
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle.BUNDLE_B64))).namelist()
    for must in ("config.py", "models/train_all.py", "orchestration/kaggle_relay/run_kaggle_session.py",
                 "requirements-models.txt", "requirements.txt", "news_pipeline/store_sync.py", "col_schema.json",
                 "common/neuralforecast_harness.py", "fusion/stacking_meta_learner.py"):
        assert must in names, must
    assert not any(n.endswith(".parquet") or "__pycache__" in n or n.startswith((".github", "output/")) for n in names)
    rec = []

    class R:
        returncode = 0

    def fake_sub_run(cmd, **kw):
        rec.append((cmd, kw)); return R()

    orig_sub_run = subprocess.run
    bundle.subprocess.run = fake_sub_run
    os.environ["IMPJR_ROOT"] = str(TMP / "kroot"); (TMP / "kroot").mkdir()
    os.environ["INSTALL_MAMBA"] = "1"
    try:
        bundle.main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        subprocess.run = orig_sub_run
    code = TMP / "kroot" / "impjr" / "feature engineering scripts"
    req = (code / "requirements-kaggle.txt").read_text()
    assert "mamba-ssm" not in req and "neuralforecast" in req and "-r requirements.txt" in req
    pip_cmd = rec[0][0]
    assert pip_cmd[1:4] == ["-m", "pip", "install"] and "-r" in pip_cmd
    if "-c" in pip_cmd:
        assert "torch==" in (code / "constraints-kaggle.txt").read_text()
    assert "mamba-ssm>=2.2" in rec[1][0]
    assert rec[-1][0][1:] == ["-m", "orchestration.kaggle_relay.run_kaggle_session"] and rec[-1][1]["cwd"] == str(code)
    # the unpacked code imports cleanly with the Kaggle env layout
    env = dict(os.environ, PROJECT_ROOT=str(TMP / "kroot" / "impjr"), PYTHONPATH=str(code))
    r = subprocess.run([sys.executable, "-c",
                        "import config, models.train_all, orchestration.kaggle_relay.run_kaggle_session, "
                        "news_pipeline.store_sync; print(config.OUTPUT_ROOT); print(config.NEWS_ARTICLES_ROOT)"],
                       cwd=str(code), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert str(TMP / "kroot" / "impjr") in r.stdout
    del os.environ["IMPJR_ROOT"]
    print(f"7 bundle ok ({size_kib:.0f} KiB, {len(names)} files)")

    # ---- 8. relay push: builds bundle + metadata, calls kaggle CLI (mocked)
    import orchestration.kaggle_relay.poll_and_relaunch as pr
    os.environ["KAGGLE_USERNAME"] = "someone"; config.KAGGLE_USERNAME = "someone"
    pushed = []

    class P:
        returncode = 0; stdout = "ok"; stderr = ""

    orig_pr_run = pr.subprocess.run
    pr.subprocess.run = lambda cmd, **kw: (pushed.append(cmd), P())[1]
    stage = TMP / "push"; stage.mkdir()
    assert pr.push_kernel(stage)
    meta = json.loads((stage / "kernel-metadata.json").read_text())
    assert meta["code_file"] == bb.BUNDLE_NAME and (stage / bb.BUNDLE_NAME).exists()
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and meta["enable_internet"]
    assert pushed[0][:4] == ["kaggle", "kernels", "push", "-p"] and pushed[0][4] == str(stage)
    pr.subprocess.run = orig_pr_run
    del os.environ["KAGGLE_USERNAME"]
    print("8 relay push ok")
finally:
    huggingface_hub.hf_hub_download = orig_dl
    hmu.get_api = orig_get_api
    oof.OOF_DIR = orig_oof_dir
    ta.MEMBERS, ta.FUSION_STAGES, ta._run, ta._cuda_count = saved_members, saved_fusion, orig_run, orig_cuda
    shutil.rmtree(TMP, ignore_errors=True)
print("ALL OK")
