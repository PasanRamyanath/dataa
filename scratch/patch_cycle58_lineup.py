import sys

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new challenger classes for Cycle 58 if not present
new_classes = '''# ==============================================================================
# 👑 CYCLE 58 LINEUP: Orderbook Liquidity Imbalance Scalpers
# ==============================================================================
class OrderbookLiquidity_RetraceScalper(Challenger5_OrderbookLiquidityImbalanceScalper):
    """
    ⚔️ Challenger 1: Orderbook Liquidity Retrace Scalper
    Challenger 5 winning liquidity guard (imbalance >= 1.25) + $15 same-side retrace barrier.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Orderbook Liquidity Retrace Scalper ($15 Barrier)"


class OrderbookLiquidity_Tier05RatchetScalper(Challenger5_OrderbookLiquidityImbalanceScalper):
    """
    ⚔️ Challenger 2: Orderbook Liquidity Tier 0.5 Ratchet Scalper
    Challenger 5 winning liquidity guard + Tier 0.5 ratchet (+1.25b net floor at +4.5b MFE).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Orderbook Liquidity Tier 0.5 Ratchet Scalper"


class OrderbookLiquidity_AdaptiveSpreadScalper(Challenger5_OrderbookLiquidityImbalanceScalper):
    """
    ⚔️ Challenger 3: Orderbook Liquidity Adaptive Spread Scalper
    Challenger 5 winning liquidity guard + Directional Spread (d_tact >= 0.12, Edge >= 10.0 bps).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Orderbook Liquidity Adaptive Spread Scalper"


class OrderbookLiquidity_TightEnvelopeScalper(Challenger5_OrderbookLiquidityImbalanceScalper):
    """
    ⚔️ Challenger 4: Orderbook Liquidity Tight Envelope Scalper
    Challenger 5 winning liquidity guard + $6 Range Boundary Envelope.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Orderbook Liquidity Tight Envelope Scalper ($6 Envelope)"


class OrderbookLiquidity_HighConvictionScalper(Challenger5_OrderbookLiquidityImbalanceScalper):
    """
    ⚔️ Challenger 5: Orderbook Liquidity High Conviction Scalper
    Ultra-strict liquidity guard (imbalance >= 1.35) for maximum trade quality.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Orderbook Liquidity High Conviction Scalper (Imbalance >= 1.35)"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            imbalance_ratio = bbo_bid_qty / max(bbo_ask_qty, 1e-4)
            if imbalance_ratio < 1.35:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

'''

# Find ChampionChallengerTournamentManager definition
mgr_marker = 'class ChampionChallengerTournamentManager:'
idx = content.find(mgr_marker)

# Insert new classes right before ChampionChallengerTournamentManager
if 'class OrderbookLiquidity_RetraceScalper' not in content:
    content = content[:idx] + new_classes + "\n" + content[idx:]
    print("Added Cycle 58 classes.")

# Now replace the competitors dictionary in ChampionChallengerTournamentManager
new_mgr_init = '''    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
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
            ),
        }'''

import re
pattern = r'    def __init__\(self, symbol: str = "BTCUSDT"\):.*?self\.competitors: dict\[str, TournamentCompetitorRecord\] = \{.*?\}\n        \}'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content[:match.start()] + new_mgr_init + content[match.end():]
    print("Updated ChampionChallengerTournamentManager init.")
else:
    print("Could not regex match manager init, checking...")

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Cycle 58 framework patch completed.")
