from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
import inspect

lines = inspect.getsource(HierarchicalSniperExecutionEngine.on_market_micro_tick).splitlines()
for i, line in enumerate(lines):
    if 'FILLED_SWING_HOLDING' in line or 'filled' in line.lower() or 'trade_through' in line.lower() or 'queue_tracker' in line:
        print(f"L{i}: {line}")
