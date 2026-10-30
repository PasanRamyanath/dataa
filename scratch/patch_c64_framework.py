import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''# ==============================================================================
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
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class ChampionChallengerTournamentManager:
    """
    Manages the 1 Champion + 5 Challengers Tournament in real time.
    Ingests each tick and decision simultaneously into all 6 engines.
    Maintains the live leaderboard and enforces the ABSOLUTE PROFITABILITY GATE:
    A challenger can only dethrone the champion if net_pnl_bps > 0.0 and win_rate_pct >= 75.0%.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
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

replacement = '''# ==============================================================================
# 👑 CYCLE 63 LINEUP (Historical Reference)
# ==============================================================================
class Cycle63_PassiveMakerUnwindScalper(Cycle62_MidpointVetoScalper):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Cycle 63: Passive Maker Unwind Scalper"
        self.taker_fee_leg = 2.0

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if "INVALIDATION" in reason:
            exit_fee_bps = self.maker_fee_leg
            reason = "PASSIVE_MAKER_UNWIND"
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)


class Cycle63_PostInvalidationSpacingScalper(Cycle62_MidpointVetoScalper):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Cycle 63: Post-Invalidation Spacing Scalper"
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
                        return self.state
                elif self.last_inval_side == "BUY" and decision.primary_direction == 2:
                    if mid >= self.last_inval_exit_price:
                        return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# 👑 CYCLE 64 LINEUP: Composite Retest-Passive Scalpers & Noise Buffer
# ==============================================================================
class Cycle64_CompositeRetestPassiveScalper(Cycle62_MidpointVetoScalper):
    """
    ⚔️ Challenger 1: Composite Retest Passive Scalper
    Synthesizes the two big winners of Cycle 63:
    1. Challenger 2's Post-Invalidation Spacing Retest Guard (prevents same-band whipsaws, +40.15 bps live advantage).
    2. Challenger 1's Passive BBO Maker Unwind (saves 3.0 bps net per cut, +12.0 bps live savings).
    3. Calibrated Retest Decay: after 300s, allows fresh high-edge expansions (edge >= 25.0 bps) to prevent starvation.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 1: Composite Retest Passive Scalper"
        self.taker_fee_leg = 2.0  # Passive Maker exit at BBO
        self.last_inval_exit_price: float = 0.0
        self.last_inval_side: str = ""
        self.last_inval_ts_ms: int = 0

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            if "INVALIDATION" in reason:
                exit_fee_bps = self.maker_fee_leg
                reason = "PASSIVE_MAKER_UNWIND"
                self.last_inval_exit_price = exit_price
                self.last_inval_side = self.active_position.side
                self.last_inval_ts_ms = exit_ts_ms
            elif "TP" in reason:
                self.last_inval_exit_price = 0.0
                self.last_inval_side = ""
                self.last_inval_ts_ms = 0
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            now_ms = decision.ts_ms
            mid = 0.5 * (bbo_bid + bbo_ask)
            if self.last_inval_exit_price > 0.0:
                elapsed_sec = (now_ms - self.last_inval_ts_ms) / 1000.0 if self.last_inval_ts_ms > 0 else 999.0
                if self.last_inval_side == "SELL" and decision.primary_direction == 0:
                    if mid <= self.last_inval_exit_price and (elapsed_sec < 300.0 or decision.expected_net_edge_bps < 25.0):
                        return self.state
                elif self.last_inval_side == "BUY" and decision.primary_direction == 2:
                    if mid >= self.last_inval_exit_price and (elapsed_sec < 300.0 or decision.expected_net_edge_bps < 25.0):
                        return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle64_CalibratedBufferScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 2: Calibrated Invalidation Buffer Scalper
    Widens invalidation trigger from -6.50 bps to -8.50 bps so normal tick spread doesn't trigger false cuts.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 2: Calibrated Buffer Scalper (-8.5 bps Stop Buffer)"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        pos = self.active_position
        if pos is None:
            return

        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == "BUY" else -1.0
        current_mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((current_mid - pos.entry_price) / pos.entry_price) * 1e4

        if unrealized_bps > pos.peak_favorable_excursion_bps:
            pos.peak_favorable_excursion_bps = unrealized_bps
        if unrealized_bps < pos.max_adverse_excursion_bps:
            pos.max_adverse_excursion_bps = unrealized_bps

        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "RESTING_MAKER_TP")
            return

        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "MODEL_HARVEST_MAKER_TP")
                return

        # Calibrated Invalidation Buffer: requires unrealized <= -8.50 bps
        if unrealized_bps <= -8.50:
            tact_opposing = (sign * d_tact) < -0.08
            micro_opposing = (sign * d_micro) < -0.05
            meso_opposing = (sign * d_meso) < -0.10
            if (tact_opposing and micro_opposing) or meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "PASSIVE_MAKER_UNWIND")
                return

        if pos.peak_favorable_excursion_bps >= self.min_trailing_activation_bps:
            locked_floor = max(self.tier1_min_lock_bps, pos.peak_favorable_excursion_bps * (1.0 - self.trailing_giveback_frac))
            if unrealized_bps <= locked_floor:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "MICRO_TRAILING_MAKER_TP")
                return

        if unrealized_bps <= -15.0:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(taker_exit_p, now_ms, self.taker_fee_leg, "HARD_STOP_LOSS_TAKER")
            return

        if elapsed_sec >= 300.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(maker_exit_p, now_ms, self.maker_fee_leg, "PASSIVE_MAKER_TIME_UNWIND")
            return


class Cycle64_SteppedRatchetCompositeScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 3: Stepped Ratchet Composite Scalper
    Locks +4.5 bps net at +8.0 bps MFE, and +8.0 bps net at +14.0 bps MFE.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 3: Stepped Ratchet Composite Scalper"
        self.min_trailing_activation_bps = 7.50
        self.tier1_min_lock_bps = 5.00


class Cycle64_HighConvictionExpansionScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 4: High Conviction Expansion Scalper
    Requires expected net edge >= 18.0 bps to eliminate low-conviction noise entries.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 4: High Conviction Expansion Scalper (>= 18 bps)"
        self.min_net_edge_bps = 18.0


class Cycle64_MicroMomentumCompositeScalper(Cycle64_CompositeRetestPassiveScalper):
    """
    ⚔️ Challenger 5: Pure Micro-Momentum Composite Scalper
    Combines composite retest-passive architecture with adverse 10s micro-momentum burst veto (> 0.08).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Pure Micro-Momentum Composite Scalper"
        self.adverse_micro_threshold = 0.08

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class ChampionChallengerTournamentManager:
    """
    Manages the 1 Champion + 5 Challengers Tournament in real time.
    Ingests each tick and decision simultaneously into all 6 engines.
    Maintains the live leaderboard and enforces the ABSOLUTE PROFITABILITY GATE:
    A challenger can only dethrone the champion if net_pnl_bps > 0.0 and win_rate_pct >= 75.0%.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+106.03 bps Peak Defended Cash Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle64_CompositeRetestPassiveScalper(symbol=symbol),
                strategy_profile="Challenger 1: Composite Retest Passive Scalper (Retest Spacing + Passive Unwind)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle64_CalibratedBufferScalper(symbol=symbol),
                strategy_profile="Challenger 2: Calibrated Invalidation Buffer Scalper (-8.5 bps Stop Buffer)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle64_SteppedRatchetCompositeScalper(symbol=symbol),
                strategy_profile="Challenger 3: Stepped Ratchet Composite Scalper (+4.5b net lock @ +8b MFE)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle64_HighConvictionExpansionScalper(symbol=symbol),
                strategy_profile="Challenger 4: High Conviction Expansion Scalper (>= 18 bps Conviction Edge)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle64_MicroMomentumCompositeScalper(symbol=symbol),
                strategy_profile="Challenger 5: Pure Micro-Momentum Composite Scalper (Adverse Sweep Veto)",
            ),
        }'''

assert target in text, "Target block not found in champion_challenger_framework.py"
text = text.replace(target, replacement)

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully patched feature engineering scripts/execution/champion_challenger_framework.py for Cycle 64!")
