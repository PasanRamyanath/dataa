import os
import json
import numpy as np

# Load cycle ledgers
base = "D:/PrivateProjects/BTC Futures/scratch"

cycles = {
    "Cycle 31 (C31)": os.path.join(base, "archive_cycle31/live_hierarchical_session_1h_ledger.jsonl"),
    "Cycle 35 (C35)": os.path.join(base, "archive_cycle35/live_hierarchical_session_1h_ledger.jsonl"),
    "Cycle 43 (C43)": os.path.join(base, "archive_cycle43/live_hierarchical_session_1h_ledger.jsonl"),
    "Cycle 60 (C60)": os.path.join(base, "archive_cycle60/live_hierarchical_session_1h_ledger.jsonl"),
    "Cycle 64 Live (C64)": os.path.join(base, "live_hierarchical_session_1h_ledger.jsonl"),
}

class PeakSynthesizedEngine:
    def __init__(self):
        # Entry parameters (Conformal confidence + High edge)
        self.min_edge_bps = 28.0
        self.tactical_softmax_gate = 0.42
        self.envelope_retrace = 6.0
        
        # Exit parameters (Multi-tier stepped ratchet + Instant breakeven + Passive maker)
        self.maker_fee = 2.0 # bps per leg (4.0 bps roundtrip)
        self.be_shield_mfe = 3.80 # bps MFE to trigger breakeven shield
        self.be_gross_floor = 4.15 # bps gross (+0.15 bps net locked)
        
        self.tier1_mfe = 6.00 # bps MFE to trigger tier 1 ratchet
        self.tier1_gross_floor = 6.25 # bps gross (+2.25 bps net locked)
        
        self.tier2_mfe = 10.00 # bps MFE to trigger runner expansion ratchet
        self.tier2_gross_floor = 8.50 # bps gross (+4.50 bps net locked)
        
        self.tp_target_bps = 35.0 # bps runner expansion target
        self.max_holding_sec = 600 # 10 minutes max holding
        self.wide_maker_buffer = 14.5 # bps passive maker exit limit (no panic taker cuts)

    def replay_cycle(self, ledger_path):
        if not os.path.exists(ledger_path):
            return None
            
        with open(ledger_path, "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]
            
        position = None
        closed_trades = []
        cumulative_pnl = 0.0
        
        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", 0.0)
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            ts_ms = bar.get("ts_ms", 0)
            
            # Position management
            if position is not None:
                pos_side = position["side"]
                entry_px = position["entry_price"]
                entry_ts = position["entry_ts"]
                duration = (ts_ms - entry_ts) / 1000.0
                
                # Calculate excursions
                if pos_side == "BUY":
                    cur_gross = (bid - entry_px) / entry_px * 10000.0
                else:
                    cur_gross = (entry_px - ask) / entry_px * 10000.0
                    
                position["mfe"] = max(position["mfe"], cur_gross)
                position["mae"] = min(position["mae"], cur_gross)
                mfe = position["mfe"]
                
                # Dynamic Ratchet Floor Evaluation
                gross_floor = None
                exit_reason = None
                
                # Check Ratchets
                if mfe >= self.tier2_mfe:
                    gross_floor = self.tier2_gross_floor
                    exit_reason = "TIER2_RUNNER_RATCHET_TP"
                elif mfe >= self.tier1_mfe:
                    gross_floor = self.tier1_gross_floor
                    exit_reason = "TIER1_STEPPED_RATCHET_TP"
                elif mfe >= self.be_shield_mfe:
                    gross_floor = self.be_gross_floor
                    exit_reason = "TIER0_BREAKEVEN_MAKER_SHIELD"
                    
                # 1. Target TP reached
                if cur_gross >= self.tp_target_bps:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "TARGET_MAKER_TP"
                    })
                    cumulative_pnl += net_pnl
                    position = None
                    continue
                    
                # 2. Ratchet Floor hit on retrace
                if gross_floor is not None and cur_gross <= gross_floor:
                    net_pnl = gross_floor - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": gross_floor, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": exit_reason
                    })
                    cumulative_pnl += net_pnl
                    position = None
                    continue
                    
                # 3. Model harvest if model flips opposite with high conviction
                opp_dir = 0 if pos_side == "BUY" else 2
                tact_probs = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                if tact_probs[opp_dir] >= 0.48 and cur_gross >= 4.0:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "MODEL_OPPOSITE_HARVEST"
                    })
                    cumulative_pnl += net_pnl
                    position = None
                    continue
                    
                # 4. Passive Maker Time Unwind if held beyond max duration and profitable
                if duration >= self.max_holding_sec and cur_gross >= 4.0:
                    net_pnl = cur_gross - (2 * self.maker_fee)
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": cur_gross, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_MAKER_TIME_UNWIND"
                    })
                    cumulative_pnl += net_pnl
                    position = None
                    continue
                    
                # 5. Wide passive maker limit (only on severe extreme breakdown)
                if cur_gross <= -self.wide_maker_buffer:
                    net_pnl = -self.wide_maker_buffer - (2 * self.maker_fee) # maker exit
                    closed_trades.append({
                        "side": pos_side, "entry": entry_px, "exit": mid,
                        "gross": -self.wide_maker_buffer, "net": net_pnl, "dur": duration,
                        "mfe": mfe, "mae": position["mae"], "reason": "PASSIVE_MAKER_WIDE_BUFFER"
                    })
                    cumulative_pnl += net_pnl
                    position = None
                    continue
                    
            # Entry evaluation (if IDLE)
            if position is None:
                p_dir = bar.get("primary_direction", 1)
                edge = bar.get("expected_net_edge_bps", 0.0)
                tact_probs = bar.get("tactical_probs", [0.33, 0.33, 0.33])
                meso_probs = bar.get("meso_probs", [0.33, 0.33, 0.33])
                
                # Check BUY Entry
                if p_dir == 2 and edge >= self.min_edge_bps:
                    if tact_probs[2] >= self.tactical_softmax_gate and meso_probs[2] >= 0.25:
                        position = {
                            "side": "BUY", "entry_price": ask, "entry_ts": ts_ms,
                            "mfe": 0.0, "mae": 0.0
                        }
                # Check SELL Entry
                elif p_dir == 0 and edge >= self.min_edge_bps:
                    if tact_probs[0] >= self.tactical_softmax_gate and meso_probs[0] >= 0.25:
                        position = {
                            "side": "SELL", "entry_price": bid, "entry_ts": ts_ms,
                            "mfe": 0.0, "mae": 0.0
                        }
                        
        wins = len([t for t in closed_trades if t["net"] > 0])
        cnt = len(closed_trades)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        avg_pnl = (cumulative_pnl / cnt) if cnt > 0 else 0.0
        
        return {
            "trades_cnt": cnt,
            "wins": wins,
            "losses": cnt - wins,
            "win_rate": wr,
            "cumulative_net_pnl": cumulative_pnl,
            "avg_net_pnl_trade": avg_pnl,
            "trades": closed_trades
        }

engine = PeakSynthesizedEngine()
print("="*100)
print("TESTING PEAK SYNTHESIZED SYSTEM ACROSS KEY HISTORICAL BENCHMARK CYCLES")
print("="*100)

for name, path in cycles.items():
    res = engine.replay_cycle(path)
    if res:
        print(f"\n>>> {name}:")
        print(f"  Trades: {res['trades_cnt']} | Wins: {res['wins']} | Losses: {res['losses']} | Win Rate: {res['win_rate']:.1f}%")
        print(f"  Cumulative Net PnL: {res['cumulative_net_pnl']:+6.2f} bps | Avg Net PnL/Trade: {res['avg_net_pnl_trade']:+6.2f} bps")
        for i, t in enumerate(res["trades"]):
            print(f"    Trade #{i+1}: {t['side']} | Net: {t['net']:+6.2f} bps | MFE: {t['mfe']:.2f}b | MAE: {t['mae']:.2f}b | Dur: {t['dur']:.0f}s | Reason: {t['reason']}")
