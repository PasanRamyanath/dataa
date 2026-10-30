from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
import inspect

lines = inspect.getsource(HierarchicalSniperExecutionEngine.on_market_micro_tick).splitlines()
for i in range(65, 86):
    print(f"L{i}: {lines[i]}")
