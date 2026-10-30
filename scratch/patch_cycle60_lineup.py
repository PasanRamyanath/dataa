import sys, os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

target_file = "feature engineering scripts/execution/champion_challenger_framework.py"

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

# We replace from class Cycle35_LiquidityShieldScalper down to the end of ChampionChallengerTournamentManager.__init__
old_block = """class Cycle35_LiquidityShieldScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 1: Cycle 35 Liquidity Shield Scalper
    Cycle 35 conformal certainty + Challenger 5 winning book depth guard (imbalance >= 1.30).
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Cycle 35 Liquidity Shield Scalper (Imbalance >= 1.30)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            imbalance_ratio = bbo_bid_qty / max(bbo_ask_qty, 1e-4)
            if imbalance_ratio < 1.30:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle35_HighConvictionLiquidityScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 2: Cycle 35 High-Conviction Liquidity Scalper
    Cycle 35 conformal certainty + ultra-strict book depth guard (imbalance >= 1.35).
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Cycle 35 High-Conviction Liquidity Scalper (Imbalance >= 1.35)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            imbalance_ratio = bbo_bid_qty / max(bbo_ask_qty, 1e-4)
            if imbalance_ratio < 1.35:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle35_TightRangeReversionScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 3: Cycle 35 Tight Range Reversion Scalper
    Cycle 35 baseline with tightened $6 Range Boundary Envelope for high volatility protection.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.strategy_profile = "Challenger 3: Cycle 35 Tight Range Reversion Scalper ($6 Envelope)"


class Cycle35_QuadHorizonResonanceScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 4: Cycle 35 Quad-Horizon Resonance Scalper
    Cycle 35 baseline + Quad-Horizon directional agreement (Micro, Tactical, Meso, Macro).
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Cycle 35 Quad-Horizon Resonance Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE" and self.ticks_seen >= 10:
            d_macro = decision.macro_probs[2] - decision.macro_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_micro = decision.micro_probs[2] - decision.micro_probs[0]
            if not ((d_micro > 0) and (d_tact > 0) and (d_meso >= 0) and (d_macro > -0.05)):
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


class Cycle35_Tier05_SteppedScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 5: Cycle 35 Tier 0.5 Stepped Ratchet Scalper
    Cycle 35 baseline + ultra-responsive Tier 0.5 ratchet (+1.25b net floor at +4.5b MFE).
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Cycle 35 Tier 0.5 Stepped Scalper"


class ChampionChallengerTournamentManager:
    \"\"\"
    Manages the 1 Champion + 5 Challengers Tournament in real time.
    Ingests each tick and decision simultaneously into all 6 engines.
    Maintains the live leaderboard and enforces the ABSOLUTE PROFITABILITY GATE:
    A challenger can only dethrone the champion if net_pnl_bps > 0.0 and win_rate_pct >= 75.0%.
    \"\"\"
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+111.63 bps Peak Cumulative Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle35_LiquidityShieldScalper(symbol=symbol),
                strategy_profile="Challenger 1: Cycle 35 Liquidity Shield Scalper (Imbalance >= 1.30)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle35_HighConvictionLiquidityScalper(symbol=symbol),
                strategy_profile="Challenger 2: Cycle 35 High-Conviction Liquidity Scalper (Imbalance >= 1.35)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle35_TightRangeReversionScalper(symbol=symbol),
                strategy_profile="Challenger 3: Cycle 35 Tight Range Reversion Scalper ($6 Envelope)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle35_QuadHorizonResonanceScalper(symbol=symbol),
                strategy_profile="Challenger 4: Cycle 35 Quad-Horizon Resonance Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle35_Tier05_SteppedScalper(symbol=symbol),
                strategy_profile="Challenger 5: Cycle 35 Tier 0.5 Stepped Scalper",
            ),
        }"""

new_block = """class Cycle35_TightRangeReversionScalper(Cycle35_ReversionPureScalper):
    \"\"\"
    ⚔️ Challenger 1: Cycle 59 Proven Leader ($6 Range Envelope Scalper)
    Cycle 59 Rank 1 winner (+3.26 bps net, PF 1.21, +14.58 bps maker win).
    Restricts entries within $6 of 15m range extremes to capture peak inflections.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.strategy_profile = "Challenger 1: Cycle 59 Winner ($6 Range Envelope Scalper)"


class Cycle60_ExhaustionAwareScalper(Cycle35_TightRangeReversionScalper):
    \"\"\"
    ⚔️ Challenger 2: Exhaustion-Aware $6 Envelope Scalper
    Takes Challenger 1's winning $6 envelope and adds a dynamic Range Exhaustion Guard:
    Prevents shorting into the bottom 20% of a wide (> $100) range,
    and prevents buying into the top 20% of a wide (> $100) range.
    Eliminates extended tail-chasing while maximizing trade execution.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Exhaustion-Aware $6 Envelope Scalper"

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        if super()._is_range_boundary_violation(primary_direction, mid):
            return True
        if len(self.rolling_15m_prices) >= 18:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= 100.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct < 0.20:
                    return True
                if primary_direction == 2 and pct > 0.80:
                    return True
        return False


class Cycle60_ActiveFrequencyScalper(Cycle35_TightRangeReversionScalper):
    \"\"\"
    ⚔️ Challenger 3: Active Frequency $5 Envelope Scalper
    Calibrated $5 envelope buffer with responsive edge hurdle (min_net_edge_bps = 3.5).
    Designed to maximize trade frequency (3-6 trades/hr) without being overly conservative.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 5.0
        self.min_net_edge_bps = 3.5
        self.strategy_profile = "Challenger 3: Active Frequency $5 Envelope Scalper"


class Cycle60_SteppedRatchetScalper(Cycle35_TightRangeReversionScalper):
    \"\"\"
    ⚔️ Challenger 4: Stepped Ratchet $6 Envelope Scalper
    Winning $6 envelope + dual-tier stepped profit locks (+1.50b net at +4.5b MFE, +3.5b net at +7.5b MFE).
    Guarantees that profitable impulse moves are locked in before market reversion.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier1_min_lock_bps = 4.25
        self.model_harvest_min_bps = 5.50
        self.strategy_profile = "Challenger 4: Stepped Ratchet $6 Envelope Scalper"


class Cycle60_DynamicAsymmetricScalper(Cycle60_ExhaustionAwareScalper):
    \"\"\"
    ⚔️ Challenger 5: Dynamic Asymmetric Scalper
    Combines the winning $6 envelope, exhaustion guard, and fast trailing locks.
    \"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.tier1_min_lock_bps = 4.00
        self.strategy_profile = "Challenger 5: Dynamic Asymmetric Scalper"


class ChampionChallengerTournamentManager:
    \"\"\"
    Manages the 1 Champion + 5 Challengers Tournament in real time.
    Ingests each tick and decision simultaneously into all 6 engines.
    Maintains the live leaderboard and enforces the ABSOLUTE PROFITABILITY GATE:
    A challenger can only dethrone the champion if net_pnl_bps > 0.0 and win_rate_pct >= 75.0%.
    \"\"\"
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle35_ReversionPureScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Cycle 35 Reversion Pure Scalper (+111.63 bps Peak Cumulative Net)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle35_TightRangeReversionScalper(symbol=symbol),
                strategy_profile="Challenger 1: Cycle 59 Winner ($6 Range Envelope Scalper)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle60_ExhaustionAwareScalper(symbol=symbol),
                strategy_profile="Challenger 2: Exhaustion-Aware $6 Envelope Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle60_ActiveFrequencyScalper(symbol=symbol),
                strategy_profile="Challenger 3: Active Frequency $5 Envelope Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle60_SteppedRatchetScalper(symbol=symbol),
                strategy_profile="Challenger 4: Stepped Ratchet $6 Envelope Scalper",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Cycle60_DynamicAsymmetricScalper(symbol=symbol),
                strategy_profile="Challenger 5: Dynamic Asymmetric Scalper",
            ),
        }"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: champion_challenger_framework.py patched for Cycle 60!")
else:
    print("ERROR: old_block not found in file! Check whitespace.")
