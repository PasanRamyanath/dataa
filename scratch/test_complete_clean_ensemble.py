import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry, _score_to_simplex
from fusion.hierarchical_fusion_v3 import HierarchicalMultiHorizonFusionEngine
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from streaming.multi_horizon_ring_buffer import MultiHorizonLiveRingBuffer

def simulate_session(ledger_file):
    bars = []
    with open(ledger_file, 'r') as f:
        for line in f:
            bars.append(json.loads(line))
    print(f"\n=======================================================")
    print(f"SIMULATING {ledger_file} ({len(bars)} bars)")
    print(f"=======================================================")

    rb = MultiHorizonLiveRingBuffer()
    reg = rb.model_registry
    fusion = HierarchicalMultiHorizonFusionEngine()
    engine = HierarchicalSniperExecutionEngine()

    # Apply our un-muzzled neural sequence prediction
    orig_predict_series = reg.tier3_itransformer.__class__.predict_from_series
    
    def unmuzzled_predict_from_series(self, closes, imbs=None, aux_bias=0.0):
        if len(closes) < 8:
            return np.array([0.33, 0.34, 0.33], dtype=np.float64)
        c = np.asarray(closes, dtype=np.float32)
        c_mean = float(np.mean(c))
        c_std = max(float(np.std(c)), c_mean * 0.0005, 1e-4)
        c_norm = (c - c_mean) / c_std

        # Run forward pass through the safetensors weights
        # We can call the original forward pass up to step_slope calculation
        # Let's compute step_slope directly:
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            if self.arch == "itransformer":
                w_emb = self.tensors["enc_embedding.value_embedding.weight"]
                b_emb = self.tensors["enc_embedding.value_embedding.bias"]
                w_proj = self.tensors["projector.weight"]
                b_proj = self.tensors["projector.bias"]
                seq_len = w_emb.shape[1]
                x_pad = np.pad(c_norm[-seq_len:], (max(0, seq_len - len(c_norm)), 0), mode="edge")
                x_t = torch.from_numpy(x_pad).view(1, seq_len)
                h = F.gelu(F.linear(x_t, w_emb, b_emb))
                if "encoder.norm.weight" in self.tensors:
                    h = F.layer_norm(h, (h.shape[-1],), self.tensors["encoder.norm.weight"], self.tensors["encoder.norm.bias"])
                traj = F.linear(h, w_proj, b_proj).view(-1).numpy()
            elif self.arch == "patchtst":
                w_in = self.tensors["model.backbone.input_layer.weight"]
                b_in = self.tensors["model.backbone.input_layer.bias"]
                w_head = self.tensors["model.head.linear.weight"]
                b_head = self.tensors["model.head.linear.bias"]
                patch_len = w_in.shape[1]
                d_model = w_in.shape[0]
                n_patches = w_head.shape[1] // d_model
                stride = 8
                req_len = (n_patches - 1) * stride + patch_len
                x_pad = np.pad(c_norm[-req_len:], (max(0, req_len - len(c_norm)), 0), mode="edge")
                x_t = torch.from_numpy(x_pad)
                patches = x_t.unfold(0, patch_len, stride)[:n_patches]
                h = F.gelu(F.linear(patches, w_in, b_in)).reshape(1, -1)
                traj = F.linear(h, w_head, b_head).view(-1).numpy()
            elif self.arch == "nhits":
                w0 = self.tensors["blocks.0.layers.0.weight"]
                b0 = self.tensors["blocks.0.layers.0.bias"]
                in_dim = w0.shape[1]
                x_pad = np.pad(c_norm[-in_dim:], (max(0, in_dim - len(c_norm)), 0), mode="edge")
                h = torch.from_numpy(x_pad).view(1, in_dim)
                for l_idx in range(8):
                    wk = f"blocks.0.layers.{l_idx}.weight"
                    bk = f"blocks.0.layers.{l_idx}.bias"
                    if wk in self.tensors:
                        w_l, b_l = self.tensors[wk], self.tensors[bk]
                        if h.shape[-1] == w_l.shape[1]:
                            h = F.relu(F.linear(h, w_l, b_l)) if l_idx < 7 else F.linear(h, w_l, b_l)
                traj = h.view(-1).numpy()
            else:
                w_grn = self.tensors.get("temporal_fusion_decoder.positionwise_grn.lin_i.weight")
                b_grn = self.tensors.get("temporal_fusion_decoder.positionwise_grn.lin_i.bias")
                if w_grn is not None and b_grn is not None:
                    in_dim = w_grn.shape[1]
                    x_pad = np.pad(c_norm[-in_dim:], (max(0, in_dim - len(c_norm)), 0), mode="edge")
                    h = F.elu(F.linear(torch.from_numpy(x_pad).view(1, in_dim), w_grn, b_grn)).view(-1).numpy()
                    traj = h
                else:
                    traj = c_norm[-8:]

        if len(traj) >= 4:
            step_slope = float(0.50 * (traj[1] - traj[0]) + 0.30 * (traj[min(3, len(traj)-1)] - traj[0]) + 0.20 * (traj[-1] - traj[0]))
        else:
            step_slope = float(traj[-1] - traj[0]) if len(traj) >= 2 else 0.0

        # Un-muzzled: neural model gets 75% primary authority!
        flow_term = float(np.mean(imbs[-4:])) * 1.2 if (imbs is not None and len(imbs) >= 4) else 0.0
        combined_score = float(np.tanh(step_slope * 1.2) * 0.75 + aux_bias * 0.25 + np.tanh(flow_term) * 0.10)
        return _score_to_simplex(combined_score, base_flat=0.54, gain=3.2)

    # Patch Sequence Runner
    reg.tier3_itransformer.__class__.predict_from_series = unmuzzled_predict_from_series

    # Clean continuous displacement
    def clean_displacement(klines_um):
        b10s = list(klines_um.get("10s", []))
        if len(b10s) < 6:
            return 0.0, 0.0, 0.0
        c_arr = np.array([float(x["close"]) for x in b10s[-120:]], dtype=np.float64)
        c_now = c_arr[-1]
        n_c = len(c_arr)
        
        w_meso = min(n_c, 90)
        eq_meso = float(np.mean(c_arr[-w_meso:]))
        std_meso = max(float(np.std(c_arr[-w_meso:])), c_now * 0.00042)
        z_meso = (c_now - eq_meso) / std_meso
        
        ret_5m = ((c_now - c_arr[-min(31, n_c)]) / c_now) * 1e4
        ret_1m = ((c_now - c_arr[-min(7, n_c)]) / c_now) * 1e4
        vel_10s = ((c_now - c_arr[-min(2, n_c)]) / c_now) * 1e4
        
        # Smooth continuous physical conviction bounded cleanly in [-0.85, +0.85]
        meso_score = float(np.clip(0.40 * z_meso + 0.60 * np.tanh(ret_5m / 8.0), -0.85, 0.85))
        tact_score = float(np.clip(np.tanh(ret_1m / 4.0), -0.85, 0.85))
        micro_score = float(np.clip(np.tanh(vel_10s / 2.0), -0.85, 0.85))
        return meso_score, tact_score, micro_score

    LiveMultiHorizonV4ModelRegistry._compute_regime_displacement = staticmethod(clean_displacement)

    # Execute simulation
    for i, b in enumerate(bars):
        mid = b["bbo_mid"]
        bid = b["bbo_bid"]
        ask = b["bbo_ask"]
        ts_ms = b["ts_ms"]
        
        rb.push_10s_bar({
            "close": mid,
            "close_ts": ts_ms,
            "open": mid,
            "high": mid + 0.5,
            "low": mid - 0.5,
            "volume": 2.0,
            "buy_volume": 1.0,
            "sell_volume": 1.0,
            "bbo_bid_qty": 1.5,
            "bbo_ask_qty": 1.5,
        })

        bar_dict = {"bbo_bid_qty": 1.5, "bbo_ask_qty": 1.5, "buy_volume": 1.0, "sell_volume": 1.0}
        micro_p, safe_fill_prob = reg.predict_tier1_micro_and_survival(rb, bar_dict)

        dec = fusion.evaluate_multi_horizon_tick(
            ts_ms=ts_ms,
            p_macro=rb.cached_macro_probs,
            p_meso=rb.cached_meso_probs,
            p_tactical=rb.cached_tactical_probs,
            p_micro=micro_p,
            atr_15m_bps=20.0,
            atr_1h_bps=25.0,
            atr_10s_bps=3.0,
            micro_safe_fill_prob=safe_fill_prob,
            hmm_regime_probs=reg.cached_hmm_probs,
        )

        fsm_pre = engine.on_market_micro_tick(
            ts_ms=ts_ms,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.5,
            bbo_ask_qty=1.5,
            aggressive_sell_vol_at_bid=1.0,
            aggressive_buy_vol_at_ask=1.0,
            eth_1s_ret_bps=0.0,
            bar_high=mid + 0.5,
            bar_low=mid - 0.5,
            current_decision=dec,
        )
        fsm_post = engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.5,
            bbo_ask_qty=1.5,
            equity_usdt=1000.0,
            intra_bar_open=mid,
        )

    closed = engine.closed_trades
    wins = sum(1 for t in closed if t.net_return_bps > 0)
    net_pnl = sum(t.net_return_bps for t in closed)
    print(f"Closed Trades: {len(closed)} | Wins: {wins} | Net PnL: {net_pnl:+.2f} bps")
    for t in closed:
        print(f"  {t.side:4s} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | net: {t.net_return_bps:+.2f} bps ({t.exit_reason})")
    if engine.active_position:
        print(f"  Still Open: {engine.active_position.side} @ {engine.active_position.entry_price:.2f}")

simulate_session('scratch/live_hierarchical_session_1h_ledger.jsonl')
