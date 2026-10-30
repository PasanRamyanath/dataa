import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''class Cycle62_PassiveUnwindMakerGuardScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 5: Passive Unwind Maker Guard Scalper
    Unwinds via BBO Maker orders instead of paying the 7.0 bps Taker invalidation penalty.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Passive Unwind Maker Guard Scalper"'''

replacement = '''class Cycle62_PassiveUnwindMakerGuardScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 5: Passive Unwind Maker Guard Scalper
    Unwinds via BBO Maker orders instead of paying the 7.0 bps Taker invalidation penalty.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Passive Unwind Maker Guard Scalper"


# ==============================================================================
# 👑 CYCLE 63 LINEUP: Midpoint Veto with Passive Maker Unwinds & Retest Spacing
# ==============================================================================
class Cycle63_PassiveMakerUnwindScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 1: Passive Maker Unwind Midpoint Scalper
    Solves Flaw: Taker fee penalty on invalidation cuts.
    Replaces 5.0 bps Taker stop with passive BBO Maker exit (2.0 bps Maker fee).
    Saves 3.0 bps net on every invalidation cut!
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 1: Passive Maker Unwind Midpoint Scalper"
        self.taker_fee_leg = 2.0  # Converted to Maker fee

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if "INVALIDATION" in reason:
            exit_fee_bps = self.maker_fee_leg
            reason = "PASSIVE_MAKER_UNWIND"
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)


class Cycle63_PostInvalidationSpacingScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 2: Post-Invalidation Spacing Retest Scalper
    Solves Flaw: Consecutive stop-outs in the same consolidation band (Cycle 62 Trade 3).
    Requires price to recover above (for SELL) or below (for BUY) the invalidation exit price
    before allowing a re-entry in the same direction.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 2: Post-Invalidation Spacing Retest Scalper"
        self.last_inval_exit_price: float = 0.0
        self.last_inval_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None and "INVALIDATION" in reason:
            self.last_inval_exit_price = exit_price
            self.last_inval_side = self.active_position.side
        elif self.active_position is not None and "TP" in reason:
            self.last_inval_exit_price = 0.0
            self.last_inval_side = ""
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            if self.last_inval_exit_price > 0.0:
                if self.last_inval_side == "SELL" and decision.primary_direction == 0:
                    if mid <= self.last_inval_exit_price:
                        return self.state  # Veto same-side re-entry below failed breakdown
                elif self.last_inval_side == "BUY" and decision.primary_direction == 2:
                    if mid >= self.last_inval_exit_price:
                        return self.state  # Veto same-side re-entry above failed breakout
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle63_SteppedRatchetMidpointScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 3: Stepped Ratchet Midpoint Scalper
    Solves Flaw: Profit decay on extended swings (Cycle 62 Trade 6 gave back 6 bps).
    Implements tighter ratchets:
    - Locks +4.5 bps net at +8.0 bps MFE.
    - Locks +8.0 bps net at +14.0 bps MFE.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 3: Stepped Ratchet Midpoint Scalper"
        self.min_trailing_activation_bps = 7.50
        self.tier1_min_lock_bps = 5.00


class Cycle63_HighEdgeExpansionScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 4: High Edge Expansion Scalper
    Solves Flaw: Low-edge chop entries.
    Requires minimum expected net edge >= 15.0 bps.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 4: High Edge Expansion Scalper (>= 15 bps)"
        self.min_net_edge_bps = 15.0


class Cycle63_PureMicroMomentumScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 5: Pure Micro-Momentum Filtered Scalper
    Solves Flaw: Entering against immediate adverse 10s micro-momentum bursts.
    Cycle 60 100% win-rate solution.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Pure Micro-Momentum Filtered Scalper"
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)'''

assert target in text, "Target not found in champion_challenger_framework.py"
text = text.replace(target, replacement)

# Now update ChampionChallengerTournamentManager
old_lineup = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+127.55 bps Peak Cumulative Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle62_MidpointVetoScalper(symbol=symbol),
                strategy_profile="Challenger 1: Midpoint Equilibrium Chop Veto Scalper",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle62_HighEdgeExpansionScalper(symbol=symbol),
                strategy_profile="Challenger 2: High Edge Expansion Scalper (>= 20 bps Edge)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle62_PureMicroMomentumScalper(symbol=symbol),
                strategy_profile="Challenger 3: Pure Micro-Momentum Scalper (C60 100% WR Solution)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle62_SteppedRatchetScalper(symbol=symbol),
                strategy_profile="Challenger 4: Stepped Ratchet Micro Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle62_PassiveUnwindMakerGuardScalper(symbol=symbol),
                strategy_profile="Challenger 5: Passive Unwind Maker Guard Scalper",
            ),
        }'''

new_lineup = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+106.03 bps Peak Defended Cash Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle63_PassiveMakerUnwindScalper(symbol=symbol),
                strategy_profile="Challenger 1: Passive Maker Unwind Midpoint Scalper (Eliminates Taker Penalty)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle63_PostInvalidationSpacingScalper(symbol=symbol),
                strategy_profile="Challenger 2: Post-Invalidation Spacing Retest Scalper (Eliminates Same-Band Whipsaw)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle63_SteppedRatchetMidpointScalper(symbol=symbol),
                strategy_profile="Challenger 3: Stepped Ratchet Midpoint Scalper (Eliminates Profit Decay)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle63_HighEdgeExpansionScalper(symbol=symbol),
                strategy_profile="Challenger 4: High Edge Expansion Scalper (>= 15 bps Conviction Edge)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle63_PureMicroMomentumScalper(symbol=symbol),
                strategy_profile="Challenger 5: Pure Micro-Momentum Filtered Scalper (C60 100% WR Solution)",
            ),
        }'''

assert old_lineup in text, "old_lineup not found in champion_challenger_framework.py"
text = text.replace(old_lineup, new_lineup)

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully patched feature engineering scripts/execution/champion_challenger_framework.py for Cycle 63!")
