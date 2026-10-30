import os

path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Marker to insert before
target_marker = "class ChampionChallengerTournamentManager:"
idx = content.find(target_marker)
if idx == -1:
    print("Could not find ChampionChallengerTournamentManager")
    exit(1)

cycle64_classes = '''# ==============================================================================
# 👑 CYCLE 64 LINEUP: True Peak Engine Synthesis (Cycle 60 Entry + Passive Maker Exit Shield)
# ==============================================================================
class Cycle64_TruePeakScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 1: Cycle 64 True Peak Scalper (Primary Certified Synthesis)
    Restores the unchoked Cycle 60 Champion entry (conformal certainty >= 0.42, $8 envelope).
    Discards all midpoint vetoes, micro-momentum gates, and spacing chokes (wrong variable).
    Eliminates premature -6.50 bps DUAL_HORIZON_INVALIDATION_TAKER panic cuts on $50 noise.
    Enforces 100% Passive Maker exits at BBO with structural invalidation only after 90s via Maker order.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 1: Cycle 64 True Peak Scalper (C60 Entry + Passive Maker Exit Shield)"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        hold_frac = elapsed_sec / max(1.0, float(pos.target_holding_seconds))

        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 0. Emergency Hard Stop Loss (only catastrophic structural collapse >= 18.0 bps)
        if unrealized_bps <= -18.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
            return

        # 1. Full Take Profit Barrier Hit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        # 2. Active Neural Model Profit Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MODEL_HARVEST_MAKER_TP")
                return

        # 3. Dynamic Trailing Profit Shield
        if unrealized_bps >= 3.80:
            if not getattr(pos, 'trailing_armed', False):
                pos.trailing_armed = True
                pos.peak_excursion = unrealized_bps
            else:
                pos.peak_excursion = max(pos.peak_excursion, unrealized_bps)
            
            pullback = pos.peak_excursion - unrealized_bps
            if pos.peak_excursion >= 6.50 and pullback >= 2.00:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="TRAILING_SHIELD_MAKER_TP")
                return

        # 4. Structural Invalidation via Passive Maker (elapsed >= 90s and hostile multi-horizon)
        if elapsed_sec >= 90.0 and unrealized_bps <= -3.00:
            meso_opposing = (sign * d_meso) < -0.10
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_STRUCTURAL_INVAL")
                return

        # 5. Profitable Time-Decay Unwind (elapsed >= 300s)
        if elapsed_sec >= 300.0 and unrealized_bps >= self.passive_unwind_min_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # 6. Maximum Holding Duration Guard (elapsed >= 600s)
        if elapsed_sec >= 600.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")


class Cycle64_PassiveMakerWideBufferScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 2: Passive Maker Wide Buffer Scalper
    Widens invalidation buffer to -12.0 bps and uses Passive Maker unwinds (0.0 taker fee).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 2: Passive Maker Wide Buffer Scalper (-12 bps Buffer + 0 Taker Fee)"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        if unrealized_bps <= -18.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
            return

        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        if elapsed_sec >= 90.0 and unrealized_bps <= -12.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_WIDE_INVAL")
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


class Cycle64_TrailingProfitShieldScalper(Cycle64_TruePeakScalper):
    """
    ⚔️ Challenger 3: Fast Trailing Profit Shield Scalper
    Arms trailing shield at +3.50 bps MFE, locking in gains faster before pullbacks.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 3: Fast Trailing Profit Shield Scalper (Armed at +3.5 bps)"
        self.min_trailing_activation_bps = 3.50


class Cycle64_SteppedRatchetEnvelopeScalper(Cycle64_TruePeakScalper):
    """
    ⚔️ Challenger 4: Stepped Ratchet Envelope Scalper
    Combines Cycle 60 $6 envelope with multi-tier stepped locks (+1.25b at +4.0b, +3.0b at +6.5b).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.range_boundary_buffer_dollars = 6.0
        self.strategy_profile = "Challenger 4: Stepped Ratchet Envelope Scalper ($6 Envelope + Stepped Locks)"


class Cycle64_HighConvictionExpansionScalper(Cycle64_TruePeakScalper):
    """
    ⚔️ Challenger 5: High Conviction Expansion Scalper
    Requires expected net edge >= 8.0 bps and conformal conviction >= 0.45.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: High Conviction Expansion Scalper (Edge >= 8.0 bps)"
        self.min_net_edge_bps = 8.0

'''

# Find competitors dict inside ChampionChallengerTournamentManager
dict_start = content.find("self.competitors: dict[str, TournamentCompetitorRecord] = {", idx)
dict_end = content.find("        }", dict_start) + len("        }")

new_competitors_dict = '''self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+106.03 bps Peak Defended Cash Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle64_TruePeakScalper(symbol=symbol),
                strategy_profile="Challenger 1: Cycle 64 True Peak Scalper (C60 Entry + Passive Maker Exit Shield)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle64_PassiveMakerWideBufferScalper(symbol=symbol),
                strategy_profile="Challenger 2: Passive Maker Wide Buffer Scalper (-12 bps Buffer + 0 Taker Fee)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle64_TrailingProfitShieldScalper(symbol=symbol),
                strategy_profile="Challenger 3: Fast Trailing Profit Shield Scalper (Armed at +3.5 bps)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle64_SteppedRatchetEnvelopeScalper(symbol=symbol),
                strategy_profile="Challenger 4: Stepped Ratchet Envelope Scalper ($6 Envelope + Stepped Locks)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle64_HighConvictionExpansionScalper(symbol=symbol),
                strategy_profile="Challenger 5: High Conviction Expansion Scalper (Edge >= 8.0 bps)",
            ),
        }'''

# Splice both
new_content = (
    content[:idx] +
    cycle64_classes +
    content[idx:dict_start] +
    new_competitors_dict +
    content[dict_end:]
)

with open(path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully updated champion_challenger_framework.py with Cycle 64 roster!")
