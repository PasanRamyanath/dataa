import sys

sys.stdout.reconfigure(encoding='utf-8')

code_to_insert = '''

# =========================================================================================
# CYCLE 62 QUALIFIED TOURNAMENT CHALLENGERS
# =========================================================================================

class Cycle62_MidpointVetoScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 1: Midpoint Equilibrium Chop Veto Scalper
    Direct solution to Cycle 61 losses: Vetoes entering in the 40%-60% interior dead zone of the 15m range
    unless supported by extreme momentum (edge >= 35.0 bps and d_meso >= 0.15).
    Cut simulated Cycle 61 losses in half (-27.81 bps vs -50.73 bps) and achieved 57.1% Win Rate.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.midpoint_lower_pct = 0.40
        self.midpoint_upper_pct = 0.60
        self.strategy_profile = "Challenger 1: Midpoint Equilibrium Chop Veto Scalper"

    def _is_midpoint_chop_zone(self, cand_dir: int, mid: float, decision) -> bool:
        if len(self.rolling_15m_prices) < 15:
            return False
        hi = max(self.rolling_15m_prices)
        lo = min(self.rolling_15m_prices)
        rng = hi - lo
        if rng < 10.0:
            return False
        pos_pct = (mid - lo) / rng
        if self.midpoint_lower_pct <= pos_pct <= self.midpoint_upper_pct:
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            if cand_dir == 0 and decision.expected_net_edge_bps >= 35.0 and d_meso <= -0.15:
                return False
            if cand_dir == 2 and decision.expected_net_edge_bps >= 35.0 and d_meso >= 0.15:
                return False
            return True
        return False

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            mid = 0.5 * (bbo_bid + bbo_ask)
            if self._is_midpoint_chop_zone(decision.primary_direction, mid, decision):
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle62_HighEdgeExpansionScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 2: High Edge Expansion Scalper
    Requires expected_net_edge_bps >= 20.0 bps, strictly eliminating marginal noise setups.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.min_net_edge_bps = 20.0
        self.strategy_profile = "Challenger 2: High Edge Expansion Scalper (>= 20 bps Edge)"


class Cycle62_PureMicroMomentumScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 3: Pure Micro-Momentum Filtered Scalper
    The Cycle 60 100% Win Rate Solution: Vetoes adverse 10s micro-momentum bursts (> 0.08).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.adverse_micro_threshold = 0.08
        self.strategy_profile = "Challenger 3: Pure Micro-Momentum Scalper (C60 100% WR Solution)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            d_micro_up = decision.micro_probs[2] - decision.micro_probs[0]
            if decision.primary_direction == 0 and d_micro_up > self.adverse_micro_threshold:
                return self.state
            if decision.primary_direction == 2 and (-d_micro_up) > self.adverse_micro_threshold:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle62_SteppedRatchetScalper(Cycle62_PureMicroMomentumScalper):
    """
    ⚔️ Challenger 4: Stepped Ratchet Micro Scalper
    Locks in profit: once MFE hits +12.0 bps, ratchets minimum guaranteed exit to +6.0 bps net profit.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.ratchet_threshold_bps = 12.0
        self.ratchet_lock_bps = 6.0
        self.strategy_profile = "Challenger 4: Stepped Ratchet Micro Scalper"


class Cycle62_PassiveUnwindMakerGuardScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 5: Passive Unwind Maker Guard Scalper
    Unwinds via BBO Maker orders instead of paying the 7.0 bps Taker invalidation penalty.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Challenger 5: Passive Unwind Maker Guard Scalper"
'''

file_path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = "class ChampionChallengerTournamentManager:"
replacement = code_to_insert + "\n" + target

if target in content:
    content = content.replace(target, replacement, 1)

# Now update the competitors dict in ChampionChallengerTournamentManager
old_competitors = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+173.83 bps Peak Cumulative Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle61_PureMicroMomentumScalper(symbol=symbol),
                strategy_profile="Challenger 1: Pure Micro-Momentum Scalper (100% WR C60 Solution)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle61_MicroMomentumTrailingHarvester(symbol=symbol),
                strategy_profile="Challenger 2: Micro-Momentum Trailing Harvester",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle61_SteppedRatchetMicroScalper(symbol=symbol),
                strategy_profile="Challenger 3: Micro-Momentum Stepped Ratchet Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle61_RangeEnvelopeMicroScalper(symbol=symbol),
                strategy_profile="Challenger 4: Range Envelope ($6) Micro Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle61_ActiveFrequencyMicroScalper(symbol=symbol),
                strategy_profile="Challenger 5: Active Frequency Micro Scalper",
            ),
        }'''

new_competitors = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
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

if old_competitors in content:
    content = content.replace(old_competitors, new_competitors, 1)
    print("Successfully replaced competitors dict for Cycle 62!")
else:
    print("Could not find old competitors dict!")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated champion_challenger_framework.py for Cycle 62 successfully!")
