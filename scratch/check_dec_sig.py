import sys
sys.path.insert(0, 'feature engineering scripts')
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
import inspect

print(inspect.signature(HierarchicalDecisionOutput))
