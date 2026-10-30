with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=SteppedRatchetOFIScalper(symbol=symbol),
                strategy_profile="Reigning Champion: Dual-Tier Stepped Ratchet Scalper (+109.63 bps Peak Cumulative Net, 100% Win Rate)",
            ),
            "Challenger_1": TournamentCompetitorRecord(
                name="Challenger_1",
                engine=Cycle48_AdaptiveSpreadScalper(symbol=symbol),
                strategy_profile="Challenger 1: Cycle 48 Adaptive Spread Scalper (Bar 50 Peak Discovery)",
            ),
            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle48_Tier05_ConformalScalper(symbol=symbol),
                strategy_profile="Challenger 2: Cycle 48 Tier 0.5 Conformal Scalper (Long-Only Pure)",
            ),
            "Challenger_3": TournamentCompetitorRecord(
                name="Challenger_3",
                engine=Cycle48_TightRangePivotScalper(symbol=symbol),
                strategy_profile="Challenger 3: Cycle 48 Tight Range Pivot Scalper ($6 Envelope)",
            ),
            "Challenger_4": TournamentCompetitorRecord(
                name="Challenger_4",
                engine=Cycle48_MicroMomentumPulseScalper(symbol=symbol),
                strategy_profile="Challenger 4: Cycle 48 Micro-Momentum Pulse Scalper (Long-Only)",
            ),
            "Challenger_5": TournamentCompetitorRecord(
                name="Challenger_5",
                engine=Challenger5_OrderbookLiquidityImbalanceScalper(symbol=symbol),
                strategy_profile="Challenger 5: Orderbook Liquidity Imbalance Scalper (Pure Liquidity Guard)",
            ),
        }'''

new_block = '''        self.competitors: dict[str, TournamentCompetitorRecord] = {
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

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Replaced competitors successfully!")
else:
    print("old_block not found exactly.")

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(content)
