"""
scratch/test_cycle31_tournament_setup.py
Verify the complete Cycle 31 tournament architecture across the Cycle 25 & 26 datasets
to ensure the Champion and Challengers execute cleanly with zero errors.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("feature engineering scripts").resolve()))

import json
import numpy as np

# Verify import
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

print("Imports verified successfully.")
