from execution.champion_challenger_framework import SteppedRatchetOFIScalper
import inspect

for cls in SteppedRatchetOFIScalper.__mro__:
    if 'on_hierarchical_decision' in cls.__dict__:
        print(f"=== {cls.__name__}.on_hierarchical_decision ===")
        print(inspect.getsource(cls.on_hierarchical_decision)[:400])
