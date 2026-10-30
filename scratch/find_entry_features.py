import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scratch.exact_c35_framework import Challenger5_ConformalConfidenceFilterScalper
from testing.run_56_cycle_historical_qualification import FullCycleLedgerReplayer, get_cycle_num
import glob

all_ledgers = sorted(
    glob.glob(str(os.path.abspath("scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl"))),
    key=get_cycle_num
)

class FeatureTrackerScalper(Challenger5_ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_features = []
        self.pending_entry_features = {}

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        # Record features at entry attempt
        mid = 0.5 * (bbo_bid + bbo_ask)
        sh = max(self.rolling_15m_prices) if self.rolling_15m_prices else mid
        sl = min(self.rolling_15m_prices) if self.rolling_15m_prices else mid
        rng = sh - sl
        pct_in_range = (mid - sl) / max(rng, 1e-4)
        
        # 5m displacement
        disp_5m = 0.0
        if len(self.recent_mid_prices) >= 30:
            disp_5m = (mid - self.recent_mid_prices[-30][1]) / mid * 10000.0
            
        # 1m displacement
        disp_1m = 0.0
        if len(self.recent_mid_prices) >= 6:
            disp_1m = (mid - self.recent_mid_prices[-6][1]) / mid * 10000.0

        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]

        self.pending_entry_features = {
            "edge": decision.expected_net_edge_bps,
            "d_tact": d_tact,
            "d_meso": d_meso,
            "d_macro": d_macro,
            "d_micro": d_micro,
            "disp_5m": disp_5m,
            "disp_1m": disp_1m,
            "pct_in_range": pct_in_range,
            "range_dollars": rng
        }
        super()._place_sniper_maker_entry(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        t = super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)
        if self.closed_trades:
            last_t = self.closed_trades[-1]
            feat = dict(self.pending_entry_features)
            feat["side"] = last_t.side
            feat["net_pnl"] = last_t.net_return_bps
            feat["mfe"] = last_t.peak_mfe_bps
            feat["is_win"] = last_t.net_return_bps > 0
            self.trade_features.append(feat)
        return t

all_feats = []
for l in all_ledgers:
    rep = FullCycleLedgerReplayer(l)
    eng = FeatureTrackerScalper(symbol="BTCUSDT")
    rep.replay_engine(eng)
    all_feats.extend(eng.trade_features)

print(f"Collected features for {len(all_feats)} trades.")
win_f = [f for f in all_feats if f["is_win"]]
loss_f = [f for f in all_feats if not f["is_win"]]

print("\n--- FEATURE COMPARISON: WINS vs LOSSES ---")
features_to_check = ["edge", "d_tact", "d_meso", "d_macro", "d_micro", "disp_5m", "disp_1m", "pct_in_range", "range_dollars"]
for k in features_to_check:
    avg_w = sum(f[k] for f in win_f) / len(win_f)
    avg_l = sum(f[k] for f in loss_f) / len(loss_f)
    print(f"  {k:15s}: Wins={avg_w:+.3f} | Losses={avg_l:+.3f}")
