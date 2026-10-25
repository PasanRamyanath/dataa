"""
Two-tier AI scoring of scraped articles (plan §9.2 / §13.4).

Tier 1 — the existing FinBERT-class classifier (inference.sentiment_scorer.
load_scorer: CryptoBERT by default, lexicon fallback). Cheap, runs on every
article, gives polarity/magnitude/confidence of the headline + lead.

Tier 2 — a small instruct LLM (config.NEWS_LLM_MODEL_NAME, Qwen2.5-3B-
Instruct by default) kept warm and called event-triggered in batches. It
reads the headline AND the body and returns structured JSON:

    relevance  0..1   how much this matters for the traded asset's price
    sentiment -1..1   expected direction of the price effect (not the tone)
    impact     0..1   expected magnitude (routine update vs market-moving)
    category          regulation | hack_exploit | etf_institutional | macro |
                      adoption_partnership | market_analysis | other
    novelty    0..1   genuinely new information vs recap/opinion/price report

Why both: a sentiment classifier scores TONE, which is the wrong target —
"SEC sues exchange" and "exchange wins SEC lawsuit" have similar tone words
and opposite price effects, and most crypto articles are price recaps
(pure noise: they describe the move the model already sees in the bars).
The LLM's relevance/novelty/impact separate signal from recap, and its
sentiment is asked as a directional price-effect judgment. The classifier
stays in the blend (config.NEWS_LLM_WEIGHT) as a variance reducer and as the
full fallback whenever the LLM is disabled, unavailable, or returns
unparseable output for an item.

T4 notes: fp16 weights (no bf16 on T4, compute capability 7.5), greedy
decoding, ~64 new tokens per item. NEWS_LLM_LOAD_IN_4BIT=1 (HF backend)
loads via bitsandbytes so a 7B instruct model also fits.

Dual-T4 throughput (config.NEWS_LLM_GPUS / NEWS_LLM_BACKEND): one full
model replica per GPU, batches split across replicas (data-parallel). A 3B
fp16 model (~6 GB) fits one 15 GB T4 with room for KV cache, and on T4s
tensor-parallel is PCIe-bound (26-44% slower than one GPU for small models
in the kaggle-vllm measurements), so replicas roughly double throughput
where TP would lose it. Backends:
  - vllm (preferred when importable): continuous batching + paged KV cache,
    dtype=float16, enforce_eager=True (CUDA-graph capture is fragile on
    T4), and JSON-schema guided decoding so every reply parses. One spawned
    process per GPU (CUDA_VISIBLE_DEVICES pinned before vllm imports).
    Not in requirements-models.txt on purpose: a plain `pip install vllm`
    replaces Kaggle's torch build. Install it without deps into a separate
    target dir instead (the kaggle-vllm recipe), e.g.
        pip install --no-deps --target /kaggle/working/vllm_pkgs vllm==<ver matching torch>
    and put that dir on PYTHONPATH.
  - hf: transformers generate, one replica per GPU in this process (one
    thread each; CUDA kernels release the GIL), left padding, articles
    length-sorted before batching to cut padding waste.

Output: config.NEWS_ARTICLES_ROOT/scored/scored_{available day}.jsonl, one
row per article (text itself not repeated), read back by
news_pipeline.features.load_scored_articles().

Usage:
    python -m news_pipeline.llm_scorer --once     # score everything pending, exit
    python -m news_pipeline.llm_scorer            # event-triggered loop, model stays warm
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_pipeline.llm_scorer")

CATEGORIES = ("regulation", "hack_exploit", "etf_institutional", "macro",
              "adoption_partnership", "market_analysis", "other")
ASSET_NAME = "Bitcoin (BTC)"

_RELEVANCE_KEYWORDS = re.compile(
    r"\b(bitcoin|btc|crypto|cryptocurrenc\w*|etf|binance|coinbase|stablecoin|tether|usdt|sec|fed|"
    r"federal reserve|inflation|cpi|rate cut|rate hike|ethereum|eth|blackrock|mining|halving|exchange)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a crypto market analyst. Judge how a news article affects the price of "
    f"{ASSET_NAME} over the next few hours. Price recaps and opinion pieces that only describe "
    "moves that already happened carry low novelty. Reply with ONLY a JSON object with keys: "
    '"relevance" (0 to 1), "sentiment" (-1 very bearish to 1 very bullish, the expected price effect, '
    'not the tone), "impact" (0 routine to 1 market-moving), "category" (one of '
    + ", ".join(CATEGORIES) + '), "novelty" (0 recap to 1 genuinely new information).'
)


def build_messages(article: dict, max_chars: int = config.NEWS_LLM_MAX_INPUT_CHARS) -> list[dict]:
    body = (article.get("text") or "").strip()
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
    user = f"Published: {article.get('published_ts') or 'unknown'}\nSource: {article.get('site', '')}\n" \
           f"Headline: {article.get('title', '')}\n"
    user += f"Article:\n{body}" if body else "Article: (headline only)"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _clamp(v, lo, hi) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(lo, min(hi, f))


def parse_llm_json(text: str) -> dict | None:
    """Tolerant parse of the LLM reply: takes the first {...} block (models
    sometimes wrap JSON in prose or ```json fences), clamps every field to
    its range, maps unknown categories to "other". None if sentiment or
    relevance is missing/unparseable (the caller then falls back to tier 1)."""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*}", "}", raw.replace("'", '"')))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    obj = {str(k).lower().strip(): v for k, v in obj.items()}
    sentiment = _clamp(obj.get("sentiment"), -1.0, 1.0)
    relevance = _clamp(obj.get("relevance"), 0.0, 1.0)
    if sentiment is None or relevance is None:
        return None
    impact = _clamp(obj.get("impact"), 0.0, 1.0)
    novelty = _clamp(obj.get("novelty"), 0.0, 1.0)
    category = str(obj.get("category", "other")).lower().strip().replace(" ", "_").replace("/", "_")
    if category not in CATEGORIES:
        category = next((c for c in CATEGORIES if c.split("_")[0] in category), "other")
    return {"relevance": relevance, "sentiment": sentiment,
            "impact": 0.5 if impact is None else impact,
            "novelty": 0.5 if novelty is None else novelty, "category": category}


# JSON schema for guided decoding (vLLM): same fields/ranges parse_llm_json
# enforces, so a guided reply always parses and never falls back to tier 1.
LLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
        "impact": {"type": "number", "minimum": 0, "maximum": 1},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["relevance", "sentiment", "impact", "category", "novelty"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Backends: each returns generate(list[messages]) -> list[str]
# ---------------------------------------------------------------------------
def visible_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_gpu_ids(setting: str = config.NEWS_LLM_GPUS, n_visible: int | None = None) -> list[int]:
    """GPU indices to put one replica on each: "auto" = all visible, "N" =
    the first N, "0"/"cpu"/"none" = [] (single default-placement model)."""
    n = visible_gpu_count() if n_visible is None else n_visible
    setting = str(setting).lower().strip()
    if n == 0 or setting in ("0", "cpu", "none", ""):
        return []
    if setting == "auto":
        return list(range(n))
    return list(range(min(int(setting), n)))


def vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def resolve_backend(setting: str = config.NEWS_LLM_BACKEND, gpu_ids: list[int] | None = None) -> str:
    """"vllm" | "hf". vLLM needs a GPU; 4-bit loading is an HF-backend
    feature (bitsandbytes), so NEWS_LLM_LOAD_IN_4BIT keeps "auto" on HF."""
    setting = str(setting).lower().strip()
    if setting in ("hf", "vllm"):
        return setting
    if gpu_ids and vllm_available() and not config.NEWS_LLM_LOAD_IN_4BIT:
        return "vllm"
    return "hf"


def load_hf_generate(model_name: str, max_new_tokens: int = 64, gpu_id: int | None = None) -> Callable:
    """HF transformers backend. gpu_id pins the whole model to that GPU
    (a replica); None keeps the old placement (device_map="auto" on GPU —
    which on 2 GPUs splits ONE model across both, pipeline style, so the
    replica path is preferred there)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {}
    if torch.cuda.is_available():
        if config.NEWS_LLM_LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = {"": gpu_id} if gpu_id is not None else "auto"
    else:
        log.warning("No GPU: %s will run on CPU in float32 — slow; fine for small live batches only", model_name)
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    device = model.device

    def generate(batch_messages: list[list[dict]]) -> list[str]:
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def _close_hf() -> None:
        nonlocal model, tok
        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        model = None
        tok = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generate.close = _close_hf  # type: ignore[attr-defined]
    log.info("Loaded tier-2 news LLM %s (hf, %s)", model_name, device)
    return generate


def vllm_sampling_params(max_new_tokens: int = 64):
    """Greedy SamplingParams with JSON-schema guided decoding, across vLLM
    API generations: StructuredOutputsParams (>=0.10.2), GuidedDecodingParams
    (0.6-0.10), else unguided (parse_llm_json still tolerates prose)."""
    from vllm import SamplingParams

    base = {"temperature": 0.0, "max_tokens": max_new_tokens}
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(**base, structured_outputs=StructuredOutputsParams(json=LLM_JSON_SCHEMA))
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=LLM_JSON_SCHEMA))
    except (ImportError, TypeError):
        log.warning("this vLLM has no guided decoding API — replies are unconstrained")
        return SamplingParams(**base)


def load_vllm_generate(model_name: str, max_new_tokens: int = 64,
                       gpu_memory_utilization: float = config.NEWS_LLM_VLLM_GPU_MEMORY_UTILIZATION) -> Callable:
    """vLLM backend on whatever single GPU CUDA_VISIBLE_DEVICES exposes."""
    from vllm import LLM

    llm = LLM(model=model_name, dtype="float16", enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=4096, seed=0)
    params = vllm_sampling_params(max_new_tokens)

    def generate(batch_messages: list[list[dict]]) -> list[str]:
        outs = llm.chat(batch_messages, params, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    log.info("Loaded tier-2 news LLM %s (vllm, CUDA_VISIBLE_DEVICES=%s)", model_name,
             os.environ.get("CUDA_VISIBLE_DEVICES"))
    return generate


def _vllm_worker_main(gpu_id: int, model_name: str, max_new_tokens: int, gpu_memory_utilization: float,
                      conn, loader: Callable | None = None) -> None:
    """Spawned per GPU. Pins the device BEFORE vllm/torch initialize CUDA,
    then serves batches over `conn`: recv list[messages] -> send
    ("ok", replies) | ("error", msg); None shuts down."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        generate = (loader or load_vllm_generate)(model_name, max_new_tokens, gpu_memory_utilization)
    except Exception as e:  # report instead of dying silently
        conn.send(("error", f"load failed on GPU {gpu_id}: {e!r}"))
        return
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            conn.send(("ok", generate(msg)))
        except Exception as e:
            conn.send(("error", repr(e)))


class VllmWorker:
    """Parent-side handle of one spawned vLLM process; callable like a
    generate fn (serialized per worker by a lock)."""

    def __init__(self, gpu_id: int, model_name: str, max_new_tokens: int = 64,
                 gpu_memory_utilization: float = config.NEWS_LLM_VLLM_GPU_MEMORY_UTILIZATION,
                 start_timeout: float = 1800.0):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # never fork a CUDA-initialized parent
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_vllm_worker_main, name=f"news-vllm-gpu{gpu_id}", daemon=True,
                                 args=(gpu_id, model_name, max_new_tokens, gpu_memory_utilization, child))
        self._proc.start()
        self._lock = threading.Lock()
        self.gpu_id = gpu_id
        if not self._conn.poll(start_timeout):
            self.close()
            raise TimeoutError(f"vLLM worker on GPU {gpu_id} did not load within {start_timeout:.0f}s")
        status, payload = self._conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(payload)

    def __call__(self, batch_messages: list[list[dict]]) -> list[str]:
        with self._lock:
            self._conn.send(batch_messages)
            status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"vLLM worker GPU {self.gpu_id}: {payload}")
        return payload

    def close(self) -> None:
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=60)
        if self._proc.is_alive():
            self._proc.terminate()


def build_generate_fns(model_name: str, max_new_tokens: int = 64, backend: str | None = None,
                       gpu_ids: list[int] | None = None) -> tuple[list[Callable], str]:
    """One generate fn per replica (+ the backend actually used). A replica
    that fails to load is skipped as long as one loads; vLLM failing
    entirely falls back to HF."""
    gpu_ids = resolve_gpu_ids() if gpu_ids is None else gpu_ids
    backend = resolve_backend(backend or config.NEWS_LLM_BACKEND, gpu_ids)
    if backend == "vllm":
        workers = []
        for g in gpu_ids or [0]:
            try:
                workers.append(VllmWorker(g, model_name, max_new_tokens))
            except Exception:
                log.exception("vLLM replica on GPU %s failed to start", g)
        if workers:
            return workers, "vllm"
        log.warning("no vLLM replica started — falling back to the HF backend")
    if not gpu_ids:
        return [load_hf_generate(model_name, max_new_tokens)], "hf"
    fns = []
    for g in gpu_ids:
        try:
            fns.append(load_hf_generate(model_name, max_new_tokens, gpu_id=g))
        except Exception:
            log.exception("HF replica on GPU %s failed to load", g)
    if not fns:
        raise RuntimeError(f"tier-2 LLM {model_name} failed to load on every GPU {gpu_ids}")
    return fns, "hf"


class LlmNewsScorer:
    """Warm-loaded instruct LLM, data-parallel over `generate_fns` (one per
    GPU replica). `generate_fn` / `generate_fns` can be injected (tests, or
    a remote inference server); otherwise replicas are built from config
    (NEWS_LLM_BACKEND, NEWS_LLM_GPUS) once at construction and kept
    resident. Call close() to free the GPUs (vLLM worker processes)."""

    def __init__(self, model_name: str = config.NEWS_LLM_MODEL_NAME, generate_fn: Callable | None = None,
                 batch_size: int = config.NEWS_LLM_BATCH_SIZE, max_new_tokens: int = 64,
                 generate_fns: list[Callable] | None = None, backend: str | None = None,
                 gpu_ids: list[int] | None = None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        if generate_fns:
            self._fns, self.backend = list(generate_fns), "injected"
        elif generate_fn is not None:
            self._fns, self.backend = [generate_fn], "injected"
        else:
            self._fns, self.backend = build_generate_fns(model_name, max_new_tokens, backend, gpu_ids)
        # vLLM batches continuously inside the engine: hand it big chunks
        self.batch_size = batch_size * 16 if self.backend == "vllm" else batch_size
        log.info("tier-2 scorer: %d replica(s), backend=%s, batch=%d", len(self._fns), self.backend, self.batch_size)

    @property
    def n_replicas(self) -> int:
        return len(self._fns)

    def _run_chunk(self, fn: Callable, chunk: list[dict]) -> list[dict | None]:
        try:
            replies = fn([build_messages(a) for a in chunk])
            if len(replies) != len(chunk):
                raise ValueError(f"backend returned {len(replies)} replies for {len(chunk)} prompts")
        except Exception:
            log.exception("tier-2 LLM batch failed; falling back to tier 1 for %d articles", len(chunk))
            replies = [""] * len(chunk)
        return [parse_llm_json(r) for r in replies]

    def score_batch(self, articles: list[dict]) -> list[dict | None]:
        """Scores in input order. Articles are length-sorted into chunks
        (less padding), chunks dealt round-robin to replicas, and each
        replica works through its share on its own thread."""
        if not articles:
            return []
        order = sorted(range(len(articles)), key=lambda i: len(articles[i].get("text") or ""))
        chunks = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        shares = [chunks[r::len(self._fns)] for r in range(len(self._fns))]
        out: list[dict | None] = [None] * len(articles)

        def run_share(r: int) -> None:
            for idx in shares[r]:
                for i, res in zip(idx, self._run_chunk(self._fns[r], [articles[i] for i in idx])):
                    out[i] = res

        if len(self._fns) == 1:
            run_share(0)
        else:
            with ThreadPoolExecutor(max_workers=len(self._fns), thread_name_prefix="llm-replica") as ex:
                list(ex.map(run_share, range(len(self._fns))))
        return out

    def close(self) -> None:
        for fn in self._fns:
            if hasattr(fn, "close"):
                fn.close()
        self._fns = []
        import gc

        gc.collect()  # HF replicas are only referenced by their generate closures
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _cutoff_dt(value: str = config.NEWS_LLM_KNOWLEDGE_CUTOFF) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt + timedelta(days=1)  # the cutoff day itself is inside the training data


def llm_may_score(article: dict, cutoff: datetime | None) -> bool:
    """False when the article predates the LLM's knowledge cutoff (look-ahead
    guard, see config.NEWS_LLM_KNOWLEDGE_CUTOFF)."""
    if cutoff is None:
        return True
    ts = article.get("published_ts") or article.get("available_ts")
    if not ts:
        return True
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff


def keyword_relevance(article: dict) -> float:
    text = f"{article.get('title') or ''} {(article.get('text') or '')[:1500]}"
    hits = len(_RELEVANCE_KEYWORDS.findall(text))
    return min(1.0, 0.2 + 0.15 * hits)


class TwoTierScorer:
    def __init__(self, tier1=None, tier2: LlmNewsScorer | None = None, llm_weight: float = config.NEWS_LLM_WEIGHT,
                 llm_cutoff: str | None = config.NEWS_LLM_KNOWLEDGE_CUTOFF):
        if tier1 is None:
            from inference.sentiment_scorer import load_scorer

            tier1 = load_scorer()
        self.tier1 = tier1
        self.tier2 = tier2
        self.llm_weight = llm_weight
        self.llm_cutoff = _cutoff_dt(llm_cutoff or "")

    def close(self) -> None:
        if self.tier2 is not None:
            self.tier2.close()

    @classmethod
    def from_config(cls) -> "TwoTierScorer":
        tier2 = None
        if config.NEWS_LLM_ENABLED:
            try:
                tier2 = LlmNewsScorer()
            except Exception:
                log.exception("tier-2 LLM %s unavailable — scoring with tier 1 only", config.NEWS_LLM_MODEL_NAME)
        return cls(tier2=tier2)

    def score(self, articles: list[dict]) -> list[dict]:
        if not articles:
            return []
        tier1_texts = [f"{a.get('title', '')}. {(a.get('text') or '')[:400]}".strip() for a in articles]
        t1 = self.tier1.score_batch(tier1_texts)
        t2: list[dict | None] = [None] * len(articles)
        if self.tier2 is not None:
            eligible = [i for i, a in enumerate(articles) if llm_may_score(a, self.llm_cutoff)]
            if len(eligible) < len(articles):
                log.info("look-ahead guard: %d article(s) predate the LLM knowledge cutoff — tier 1 only",
                         len(articles) - len(eligible))
            for i, res in zip(eligible, self.tier2.score_batch([articles[i] for i in eligible])):
                t2[i] = res
        rows = []
        for a, s1, s2 in zip(articles, t1, t2):
            sent_t1 = float(s1["polarity"])
            if s2 is not None:
                sentiment = self.llm_weight * s2["sentiment"] + (1 - self.llm_weight) * sent_t1
                relevance, impact, novelty, category = s2["relevance"], s2["impact"], s2["novelty"], s2["category"]
            else:
                sentiment = sent_t1
                relevance, impact, novelty, category = keyword_relevance(a), float(s1["magnitude"]), 0.5, "other"
            rows.append({
                "article_id": a["article_id"],
                "site": a.get("site"),
                "title": a.get("title", ""),
                "published_ts": a.get("published_ts"),
                "available_ts": a["available_ts"],
                "body_available": bool(a.get("body_available")),
                "sent_tier1": sent_t1,
                "mag_tier1": float(s1["magnitude"]),
                "conf_tier1": float(s1["confidence"]),
                "llm_ok": s2 is not None,
                "sent_llm": s2["sentiment"] if s2 is not None else None,
                "sentiment": max(-1.0, min(1.0, sentiment)),
                "relevance": relevance,
                "impact": impact,
                "novelty": novelty,
                "category": category,
            })
        return rows


# ---------------------------------------------------------------------------
# Pending-work bookkeeping (raw store -> scored store)
# ---------------------------------------------------------------------------
def _raw_root() -> Path:
    return config.NEWS_ARTICLES_ROOT / "raw"


def _scored_root() -> Path:
    return config.NEWS_ARTICLES_ROOT / "scored"


def _scored_ids(scored_root: Path) -> set[str]:
    ids = set()
    for path in scored_root.glob("scored_*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["article_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def pending_articles(raw_root: Path | None = None, scored_root: Path | None = None) -> list[dict]:
    raw_root, scored_root = raw_root or _raw_root(), scored_root or _scored_root()
    done = _scored_ids(scored_root)
    pending = []
    for path in sorted(raw_root.glob("articles_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("article_id") not in done:
                    # a second scraper process (backfill + live) can append
                    # the same article; score it once or its weight doubles
                    done.add(rec.get("article_id"))
                    pending.append(rec)
    pending.sort(key=lambda r: r["available_ts"])
    return pending


def score_pending(scorer: TwoTierScorer, raw_root: Path | None = None, scored_root: Path | None = None,
                  chunk: int = 512) -> int:
    scored_root = scored_root or _scored_root()
    pending = pending_articles(raw_root, scored_root)
    if not pending:
        return 0
    scored_root.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(pending), chunk):
        rows = scorer.score(pending[i:i + chunk])
        by_day: dict[str, list[dict]] = {}
        for r in rows:
            by_day.setdefault(r["available_ts"][:10], []).append(r)
        for day, day_rows in by_day.items():
            with (scored_root / f"scored_{day}.jsonl").open("a", encoding="utf-8") as fh:
                for r in day_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("scored %d/%d pending articles", min(i + chunk, len(pending)), len(pending))
    return len(pending)


def run_forever(poll_seconds: float = 5.0, max_cycles: int | None = None, scorer: TwoTierScorer | None = None) -> None:
    """Event-triggered: the (warm) models run only when the raw store has
    grown since the last pass."""
    scorer = scorer or TwoTierScorer.from_config()
    last_sig = None
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        sig = tuple((p.name, p.stat().st_size) for p in sorted(_raw_root().glob("articles_*.jsonl")))
        if sig != last_sig:
            try:
                score_pending(scorer)
                last_sig = sig
            except Exception:
                log.exception("scoring pass failed; will retry")
        cycle += 1
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-llm", action="store_true", help="tier-1 classifier only")
    args = parser.parse_args()
    scorer = TwoTierScorer() if args.no_llm else TwoTierScorer.from_config()
    try:
        if args.once:
            n = score_pending(scorer)
            log.info("scored %d articles", n)
        else:
            run_forever(args.poll_seconds, scorer=scorer)
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
