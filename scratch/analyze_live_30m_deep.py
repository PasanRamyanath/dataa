"""
Deep Post-Run Analytical Suite for 30-Minute Live Streaming Evaluation Ledger.
Processes 193 continuous 10-second bars from Kaggle Dual-T4 VM.
Computes multi-horizon forward returns directly across the sequential bar array.
"""
import json
import numpy as np
from pathlib import Path

LEDGER_PATH = Path("scratch/live_30m_eval_ledger.jsonl")

def main():
    bars = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))

    total_bars = len(bars)
    print("=" * 80)
    print(f"DEEP AUDIT OF 30-MINUTE LIVE EVALUATION DATASET ({total_bars} BARS)")
    print("=" * 80)

    # Precompute forward returns in array
    horizons = [1, 2, 3, 6, 12, 18]
    for i, b in enumerate(bars):
        p0 = b["close_price"]
        b["fwd_rets"] = {}
        for h in horizons:
            if i + h < total_bars:
                ph = bars[i + h]["close_price"]
                if p0 > 0:
                    ret_bps = ((ph - p0) / p0) * 10000.0
                    b["fwd_rets"][h] = ret_bps

    # 1. Multi-Horizon Directional Accuracy
    dir_bars = [b for b in bars if b.get("primary_direction") in (0, 2)]
    print(f"\n1. MULTI-HORIZON DIRECTIONAL ACCURACY")
    print(f"Total Bars: {total_bars} | Directional Calls (Long/Short): {len(dir_bars)} ({len(dir_bars)/total_bars*100:.1f}%) | Chop/Flat Calls: {total_bars - len(dir_bars)} ({(total_bars - len(dir_bars))/total_bars*100:.1f}%)")
    print("-" * 75)
    print(f"{'Horizon':<12} | {'Time Span':<12} | {'Evaluated':<10} | {'Correct':<10} | {'Accuracy (%)':<12} | {'Avg Edge (bps)':<14}")
    print("-" * 75)
    for h in horizons:
        valid_bars = [b for b in dir_bars if h in b["fwd_rets"]]
        if not valid_bars:
            continue
        correct = 0
        edges = []
        for b in valid_bars:
            ret = b["fwd_rets"][h]
            pdir = b["primary_direction"]
            dir_mult = 1.0 if pdir == 2 else -1.0
            edge = dir_mult * ret
            edges.append(edge)
            if edge > 0:
                correct += 1
        acc = (correct / len(valid_bars)) * 100.0
        avg_edge = np.mean(edges)
        sec = h * 10
        print(f"H{h} ({sec:2d}s)     | {sec:3d}s ahead    | {len(valid_bars):<10} | {correct:<10} | {acc:.2f}%      | {avg_edge:+.2f} bps")

    # 2. Member Model Performance Comparison (3-Bar Horizon = 30s)
    print("\n2. MEMBER MODEL LEADERBOARD (3-Bar / 30-Second Ahead Directional Accuracy)")
    print("-" * 80)
    print(f"{'Model Name':<20} | {'Directional Calls':<18} | {'Correct':<10} | {'3-Bar Acc (%)':<12} | {'Avg Edge (bps)':<14}")
    print("-" * 80)
    models = ["ssm-challenger", "itransformer", "nhits", "kronos-finetune", "patchtst", "sentiment-branch", "gbm-family", "tft"]
    for m in models:
        m_calls = 0
        m_correct = 0
        m_edges = []
        for b in bars:
            preds = b.get("member_predictions", {})
            ret_3b = b["fwd_rets"].get(3)
            if m in preds and ret_3b is not None:
                p = preds[m]
                if len(p) == 3:
                    # Argmax: 0=down, 1=flat, 2=up
                    m_dir = int(np.argmax(p))
                    if m_dir in (0, 2):
                        m_calls += 1
                        dir_mult = 1.0 if m_dir == 2 else -1.0
                        edge = dir_mult * ret_3b
                        m_edges.append(edge)
                        if edge > 0:
                            m_correct += 1
        acc = (m_correct / max(1, m_calls)) * 100.0
        avg_edge = np.mean(m_edges) if m_edges else 0.0
        print(f"{m:<20} | {m_calls:<18} | {m_correct:<10} | {acc:.2f}%      | {avg_edge:+.2f} bps")

    # 3. Ultra Sniper Conviction Filter Audit (act_probability >= 0.70)
    print("\n3. ULTRA SNIPER CONVICTION FILTER ANALYSIS")
    print("-" * 75)
    auth_bars = [b for b in bars if b.get("is_authorized")]
    veto_bars = [b for b in bars if not b.get("is_authorized")]
    print(f"Total Bars Evaluated:           {total_bars}")
    print(f"High-Conviction Signals (>=0.70): {len(auth_bars)} ({len(auth_bars)/total_bars*100:.1f}%)")
    print(f"Chop / Low-Conviction Vetoed:   {len(veto_bars)} ({len(veto_bars)/total_bars*100:.1f}%)")
    
    for h in [1, 3, 6]:
        auth_rets = [ (1.0 if b['primary_direction']==2 else -1.0) * b['fwd_rets'][h] for b in auth_bars if h in b['fwd_rets']]
        veto_rets = [ (1.0 if b['primary_direction']==2 else -1.0) * b['fwd_rets'][h] for b in veto_bars if h in b['fwd_rets'] and b['primary_direction'] in (0, 2)]
        sec = h * 10
        print(f"\n  Horizon H{h} ({sec}s):")
        if auth_rets:
            print(f"    Authorized Signals: Acc = {np.mean([x>0 for x in auth_rets])*100:.1f}% | Mean Edge = {np.mean(auth_rets):+.2f} bps")
        if veto_rets:
            print(f"    Vetoed Chop Calls:  Acc = {np.mean([x>0 for x in veto_rets])*100:.1f}% | Mean Edge = {np.mean(veto_rets):+.2f} bps")

    # 4. Latency Telemetry Audit
    print("\n4. INFERENCE LATENCY PROFILING (Dual Tesla T4 GPUs vs 10s Bar Deadline)")
    print("-" * 75)
    latencies = [b.get("latency_telemetry", {}).get("total_turnaround_ms") for b in bars if b.get("latency_telemetry")]
    gpu_times = [b.get("latency_telemetry", {}).get("gpu_inference_ms") for b in bars if b.get("latency_telemetry")]
    feat_times = [b.get("latency_telemetry", {}).get("feature_engineering_ms") for b in bars if b.get("latency_telemetry")]
    fuse_times = [b.get("latency_telemetry", {}).get("cpu_fusion_ms") for b in bars if b.get("latency_telemetry")]
    
    latencies = [x for x in latencies if x is not None]
    if latencies:
        print(f"Feature Engineering: Mean={np.mean(feat_times):.1f}ms | p95={np.percentile(feat_times, 95):.1f}ms | Max={np.max(feat_times):.1f}ms")
        print(f"GPU Model Inference: Mean={np.mean(gpu_times):.1f}ms | p95={np.percentile(gpu_times, 95):.1f}ms | Max={np.max(gpu_times):.1f}ms")
        print(f"CPU Fusion Pipeline: Mean={np.mean(fuse_times):.1f}ms | p95={np.percentile(fuse_times, 95):.1f}ms | Max={np.max(fuse_times):.1f}ms")
        print(f"Total Turnaround:    Mean={np.mean(latencies):.1f}ms | p95={np.percentile(latencies, 95):.1f}ms | Max={np.max(latencies):.1f}ms")
        print(f"Safety Headroom:     {10000.0 - np.mean(latencies):.1f}ms ({ (10000.0 - np.mean(latencies))/1000.0:.2f}s of 10.0s budget)")

    # 5. Probability Calibration / Brier Score
    print("\n5. CALIBRATION & PROBABILITY RESOLUTION")
    print("-" * 75)
    brier_scores = []
    for b in bars:
        ret_3b = b["fwd_rets"].get(3)
        if ret_3b is not None:
            # True label: 0 if ret <= -8 bps, 2 if ret >= 8 bps, else 1
            true_label = 2 if ret_3b >= 8.0 else (0 if ret_3b <= -8.0 else 1)
            target_vec = np.zeros(3)
            target_vec[true_label] = 1.0
            pred_vec = np.array([b.get("prob_down", 0.33), b.get("prob_flat", 0.34), b.get("prob_up", 0.33)])
            brier_scores.append(np.sum((pred_vec - target_vec) ** 2))
    
    if brier_scores:
        mean_brier = np.mean(brier_scores)
        print(f"Multi-Class Brier Score: {mean_brier:.4f} (Uninformative Baseline: 0.6667 | Lower is better)")

if __name__ == "__main__":
    main()
