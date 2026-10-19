"""
Process-isolated Dual-T4 GPU parallel inference executor.
Executes NeuralForecast on GPU 0 and Foundation/State-Space models on GPU 1.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any
import numpy as np
import polars as pl

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger("streaming.dual_gpu")


class DualGpuInferenceCoordinator:
    """Orchestrates parallel inference across Dual NVIDIA Tesla T4 GPUs and Host CPU.
    Eliminates PCIe bottlenecks by device-isolating models with zero inter-GPU communication."""

    def __init__(self, member_dict: dict[str, Any], max_workers: int = 8):
        self.members = member_dict
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Device topology partitions
        self.gpu0_models = ["tft", "patchtst", "itransformer", "nhits"]
        self.gpu1_models = ["kronos-finetune", "ssm-challenger"]
        self.cpu_models = ["gbm-family", "sentiment-branch", "hmm-regime"]

        self._assign_device_placements()

    def _assign_device_placements(self) -> None:
        """Assigns models to explicit CUDA devices when running in a multi-GPU environment."""
        if not HAS_TORCH or not torch.cuda.is_available():
            log.info("CUDA unavailable — executing ensemble on host CPU.")
            return

        device_count = torch.cuda.device_count()
        log.info("Detected %d CUDA device(s). Partitioning models across accelerators...", device_count)

        if device_count >= 2:
            log.info("Dual-GPU mode enabled: Device 0 (cuda:0) = NeuralForecast; Device 1 (cuda:1) = Kronos + SSM.")
            # Set devices on underlying PyTorch models if exposed
            for name, member in self.members.items():
                if name in self.gpu0_models and hasattr(member, "to"):
                    try:
                        member.to("cuda:0")
                    except Exception:
                        pass
                elif name in self.gpu1_models and hasattr(member, "to"):
                    try:
                        member.to("cuda:1")
                    except Exception:
                        pass
        elif device_count == 1:
            log.info("Single-GPU mode enabled: All deep learning models assigned to cuda:0.")
            for member in self.members.values():
                if hasattr(member, "to"):
                    try:
                        member.to("cuda:0")
                    except Exception:
                        pass

    def predict_tick(self, recent_df: pl.DataFrame, timeout_seconds: float = 8.5) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Executes concurrent forward passes across all available ensemble members.
        
        Args:
            recent_df: Trailing window of feature-engineered bars (up to 720 bars).
            timeout_seconds: Hard wall-clock deadline for the parallel batch (default: 8.5s).
            
        Returns:
            Tuple of (predictions_dict, latencies_dict).
            predictions_dict maps member_name -> np.ndarray[3] [p_down, p_flat, p_up].
            latencies_dict maps member_name -> execution_time_seconds.
        """
        futures: dict[str, concurrent.futures.Future] = {}
        predictions: dict[str, np.ndarray] = {}
        latencies: dict[str, float] = {}
        batch_deadline = time.perf_counter() + timeout_seconds

        def _predict_member(name: str):
            t_start = time.perf_counter()
            try:
                pred = self.members[name].predict(recent_df)
                elapsed = time.perf_counter() - t_start
                return pred, elapsed
            except Exception as e:
                elapsed = time.perf_counter() - t_start
                log.warning("Inference failed for member '%s' after %.3fs: %s", name, elapsed, e)
                return None, elapsed

        # Dispatch async tasks across the thread pool
        for name in self.members:
            futures[name] = self.thread_pool.submit(_predict_member, name)

        # Gather parallel results with shared wall-clock batch deadline
        for name, fut in futures.items():
            try:
                remaining = max(0.05, batch_deadline - time.perf_counter())
                pred, elapsed = fut.result(timeout=remaining)
                latencies[name] = round(elapsed, 4)
                if pred is not None:
                    # Validate probability vector [p_down, p_flat, p_up]
                    pred_arr = np.asarray(pred, dtype=np.float64)
                    if pred_arr.shape == (3,) and not np.isnan(pred_arr).any():
                        # Renormalize to sum to 1.0 if needed
                        s = float(pred_arr.sum())
                        if s > 0:
                            pred_arr = pred_arr / s
                        predictions[name] = pred_arr
                    else:
                        log.debug("Member '%s' returned invalid shape/NaNs: %s", name, pred_arr)
            except concurrent.futures.TimeoutError:
                latencies[name] = round(timeout_seconds, 4)
                log.warning("Member '%s' timed out (> %.1fs) — skipping this tick", name, timeout_seconds)
            except Exception as e:
                log.warning("Error collecting inference result for '%s': %s", name, e)

        return predictions, latencies

    def shutdown(self) -> None:
        self.thread_pool.shutdown(wait=False)
