"""
Named Cloudflare Tunnel process management (§14.1). A *named* tunnel (not a
Quick Tunnel) is required because the plan needs a stable hostname across
every Kaggle session restart (~12h ceiling) — Quick Tunnels hand out a
random subdomain per run, carry no SLA, cap concurrency at 200, and don't
support SSE, so none of them fit here.

cloudflared makes an outbound-only connection to Cloudflare's edge, so no
inbound firewall/port-forward is needed on the Kaggle VM. This module:
  1. Downloads the official linux/amd64 cloudflared binary if not already
     present at config.CLOUDFLARED_BINARY_PATH.
  2. Reads the tunnel token via kaggle_secrets.UserSecretsClient at runtime
     (mirroring config.get_hf_token()'s exact discipline), falling back to
     the CLOUDFLARE_TUNNEL_TOKEN_ENV_FALLBACK env var only for local/offline
     testing — never hardcoded, never committed.
  3. Launches `cloudflared tunnel run --token <TOKEN>` as a background
     subprocess and exposes start/stop/is_running/restart-on-death helpers.

Never downloads the real binary or opens a real tunnel in this codebase's
own tests — every test monkeypatches subprocess.run/Popen and the
downloader function.

Usage:
    from streaming.cloudflared_manager import CloudflaredManager
    mgr = CloudflaredManager()
    mgr.ensure_binary()
    mgr.start()
    ...
    mgr.stop()

    python -m streaming.cloudflared_manager --status   # local/offline status check only
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import stat
import subprocess
import time
import urllib.request
from pathlib import Path

import config

log = logging.getLogger("streaming.cloudflared_manager")


def get_tunnel_token() -> str | None:
    """Reads the Cloudflare tunnel token from Kaggle Secrets at runtime,
    falling back to an env var for local/offline testing. Mirrors
    config.get_hf_token()'s exact pattern. Returns None if unavailable
    anywhere (caller must treat this as "tunnel unavailable, skip it" —
    never crash the training/inference run over a missing tunnel token)."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret("CLOUDFLARE_TUNNEL_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get(config.CLOUDFLARE_TUNNEL_TOKEN_ENV_FALLBACK)


def download_binary(dest_path: Path | None = None, url: str | None = None) -> Path:
    """Downloads the official cloudflared linux/amd64 binary to dest_path
    and makes it executable. Real network I/O — never called in this
    codebase's own tests, which monkeypatch this function entirely."""
    dest = Path(dest_path) if dest_path is not None else config.CLOUDFLARED_BINARY_PATH
    src_url = url if url is not None else config.CLOUDFLARED_DOWNLOAD_URL
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading cloudflared from %s to %s", src_url, dest)
    urllib.request.urlretrieve(src_url, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dest


class CloudflaredManager:
    def __init__(
        self,
        binary_path: Path | None = None,
        download_fn=download_binary,
        popen_fn=subprocess.Popen,
        token_fn=get_tunnel_token,
    ):
        self.binary_path = Path(binary_path) if binary_path is not None else config.CLOUDFLARED_BINARY_PATH
        self._download_fn = download_fn
        self._popen_fn = popen_fn
        self._token_fn = token_fn
        self._proc: subprocess.Popen | None = None

    def ensure_binary(self) -> Path:
        if self.binary_path.exists() or shutil.which(str(self.binary_path)):
            log.info("cloudflared binary already present at %s", self.binary_path)
            return self.binary_path
        return self._download_fn(self.binary_path)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """Starts `cloudflared tunnel run --token <TOKEN>` as a background
        process. Returns False (and logs, never raises) if no token is
        available or the binary is missing and can't be fetched — the
        caller (inference glue) must run fine without a tunnel."""
        if self.is_running():
            log.info("cloudflared already running (pid=%s)", self._proc.pid)
            return True

        token = self._token_fn()
        if not token:
            log.warning("No Cloudflare tunnel token available — skipping tunnel start")
            return False

        try:
            self.ensure_binary()
        except Exception:
            log.exception("Failed to obtain cloudflared binary — skipping tunnel start")
            return False

        try:
            self._proc = self._popen_fn(
                [str(self.binary_path), "tunnel", "run", "--token", token],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except Exception:
            log.exception("Failed to launch cloudflared process — skipping tunnel start")
            self._proc = None
            return False

        log.info("cloudflared tunnel started (pid=%s)", self._proc.pid)
        return True

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        log.info("cloudflared tunnel stopped")
        self._proc = None

    def restart_if_dead(self) -> bool:
        """Call periodically (e.g. from the live loop's own tick or a
        watchdog thread) to relaunch cloudflared if it died mid-session
        without killing the whole Kaggle kernel. Returns True if a
        (re)start was attempted."""
        if self.is_running():
            return False
        log.warning("cloudflared not running — attempting restart")
        return self.start()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="print token/binary availability, no network calls")
    args = parser.parse_args()
    if args.status:
        mgr = CloudflaredManager()
        token = mgr._token_fn()
        print(f"token available: {bool(token)}")
        print(f"binary present at {mgr.binary_path}: {mgr.binary_path.exists()}")
        print(f"is_running: {mgr.is_running()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
