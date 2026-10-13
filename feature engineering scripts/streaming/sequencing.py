"""
Sequence-number + session-ID stamping for every streamed payload (§14.2:
"attach a monotonically increasing sequence number and session ID to every
payload; persist the counter... so a fresh session resumes numbering rather
than resetting to zero and confusing the receiver about gaps vs.
duplicates"). The counter and session ID live in one small local JSON state
file under config.STREAMING_STATE_ROOT, read once at process start and
rewritten after every increment, so a process restart *within the same
Kaggle session* (cloudflared or the FastAPI server crashing and being
relaunched by the same notebook cell) resumes numbering instead of
resetting to zero, exactly as §14.2 asks.

Caveat, stated explicitly rather than silently assumed away (mirroring
kaggle_run_pipeline.py's own documented distinction between "this run" and
"this Kaggle session"): a full kernel relaunch via orchestration/kaggle_relay/
starts from a *fresh* Kaggle VM whose local disk does not carry over —
there is no external always-on store this layer can consult instead (unlike
the data pipeline's HF-repo-backed resumability, or the paper-trading
engine's, both of which are subject to the very same limitation whenever
their state root lives under /kaggle/working). A relaunch therefore starts a
NEW session_id at sequence 0. This is by design, not an oversight: §14.2's
own wording ("gaps vs. duplicates") implies a receiver keys off
(session_id, sequence) pairs, so a new session_id after a relaunch reads as
a fresh stream, never as a gap in the old one.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import config

log = logging.getLogger("streaming.sequencing")

STATE_FILENAME = "sequence_state.json"


class Sequencer:
    def __init__(self, state_root: Path | None = None):
        self.state_root = Path(state_root) if state_root is not None else config.STREAMING_STATE_ROOT
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_root / STATE_FILENAME
        self.session_id, self._next_seq = self._load_or_init()

    def _load_or_init(self) -> tuple[str, int]:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                return str(state["session_id"]), int(state["next_sequence"])
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                log.warning("Corrupt sequence state at %s — starting a fresh session", self.state_path)
        session_id = uuid.uuid4().hex
        self._save(session_id, 0)
        return session_id, 0

    def _save(self, session_id: str, next_seq: int) -> None:
        self.state_path.write_text(
            json.dumps({"session_id": session_id, "next_sequence": next_seq}), encoding="utf-8"
        )

    def next(self) -> tuple[str, int]:
        """Returns (session_id, sequence) for one payload and persists the
        incremented counter immediately, so a crash right after this call
        never replays the same sequence number on restart."""
        seq = self._next_seq
        self._next_seq += 1
        self._save(self.session_id, self._next_seq)
        return self.session_id, seq

    def stamp(self, payload: dict) -> dict:
        session_id, seq = self.next()
        return {"session_id": session_id, "sequence": seq, **payload}
