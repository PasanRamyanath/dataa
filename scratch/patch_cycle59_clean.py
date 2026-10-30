with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    content = f.read()

c59_classes = '''# ==============================================================================
# 👑 CYCLE 59 LINEUP: Cycle 35 Reversion Protocol with Orderbook Liquidity Shield
# ==============================================================================
class Cycle35_ReversionPureScalper(ConformalConfidenceFilterScalper):
    """
    👑 Reigning Champion: Cycle 35 Reversion Pure Scalper
    Exact restored Cycle 35 baseline (+44.64 bps net, 100% WR):
    - Conformal certainty: max(p_down, p_up) >= 0.45 across Tactical & Meso.
    - $10 Range Boundary Envelope Filter.
    - $10 Same-Side Retrace Barrier.
    - Dual-Tier Stepped Ratchet (+1.25b net at +4.5b, +2.0b net at +6.0b MFE).
    - 300s Passive Maker Time Unwind.
    - Universal IDLE Gate protection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Cycle 35 Reversion Pure Scalper (+111.63 bps Peak Cumulative Net)"


class Cycle35_LiquidityShieldScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 1: Cycle 35 Liquidity Shield Scalper
    Cycle 35 conformal certainty + Challenger 5 winning book depth guard (imbalance >= 1.30).
    """
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
    """
    ⚔️ Challenger 2: Cycle 35 High-Conviction Liquidity Scalper
    Cycle 35 conformal certainty + ultra-strict book depth guard (imbalance >= 1.35).
    """
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
    """
    ⚔️ Challenger 3: Cycle 35 Tight Range Reversion Scalper
    Cycle 35 baseline with tightened $6 Range Boundary Envelope for high volatility protection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.strategy_profile = "Challenger 3: Cycle 35 Tight Range Reversion Scalper ($6 Envelope)"


class Cycle35_QuadHorizonResonanceScalper(Cycle35_ReversionPureScalper):
    """
    ⚔️ Challenger 4: Cycle 35 Quad-Horizon Resonance Scalper
    Cycle 35 baseline + Quad-Horizon directional agreement (Micro, Tactical, Meso, Macro).
    """
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
    """
    ⚔️ Challenger 5: Cycle 35 Tier 0.5 Stepped Ratchet Scalper
    Cycle 35 baseline + ultra-responsive Tier 0.5 ratchet (+1.25b net floor at +4.5b MFE).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Cycle 35 Tier 0.5 Stepped Scalper"

'''

# Insert classes before class ChampionChallengerTournamentManager:
mgr_pos = content.find('class ChampionChallengerTournamentManager:')
content = content[:mgr_pos] + c59_classes + "\n" + content[mgr_pos:]

old_competitors = '''            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Challenger5_OrderbookLiquidityImbalanceScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Orderbook Liquidity Imbalance Scalper (+111.63 bps Peak Cumulative Net, 100% Win Rate)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=OrderbookLiquidity_RetraceScalper(symbol=symbol),
                strategy_profile="Challenger 1: Orderbook Liquidity Retrace Scalper ($15 Barrier)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=OrderbookLiquidity_Tier05RatchetScalper(symbol=symbol),
                strategy_profile="Challenger 2: Orderbook Liquidity Tier 0.5 Ratchet Scalper",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=OrderbookLiquidity_AdaptiveSpreadScalper(symbol=symbol),
                strategy_profile="Challenger 3: Orderbook Liquidity Adaptive Spread Scalper",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=OrderbookLiquidity_TightEnvelopeScalper(symbol=symbol),
                strategy_profile="Challenger 4: Orderbook Liquidity Tight Envelope Scalper ($6 Envelope)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=OrderbookLiquidity_HighConvictionScalper(symbol=symbol),
                strategy_profile="Challenger 5: Orderbook Liquidity High Conviction Scalper (Imbalance >= 1.35)",
            ),'''

new_competitors = '''            "Champion": TournamentCompetitorRecord(
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
            ),'''

if old_competitors in content:
    content = content.replace(old_competitors, new_competitors)
    print("Replaced competitors successfully!")
else:
    print("old_competitors string not found exactly.")

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
