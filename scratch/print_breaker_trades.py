import os, glob, json

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
from test_integrated_peak_engine import *

# Let's inspect which trades hit HARD_STOP_LOSS_TAKER
eng = test_fully_integrated_peak_architecture
# Let's modify test_integrated_peak_engine to print the breaker trades
