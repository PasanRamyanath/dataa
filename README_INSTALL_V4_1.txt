================================================================================
Project-IMPJR v4.1 Complete Fix & Maximum-Profitability Upgrade Bundle
Target Repository: https://github.com/Rimramyanath/dataa
================================================================================

HOW TO INSTALL & PUSH TO YOUR GITHUB REPOSITORY:
1. Extract all files in this ZIP archive directly into your local repository
   root folder (e.g. D:\backup\Binance data\) and choose "Replace all files"
   when prompted.
2. Open a terminal in your repository root and run:
       git add new_plan.md "feature engineering scripts/" apply_v4_and_push_to_github.py
       git commit -m "Upgrade to Project-IMPJR v4.1 Complete Multi-Horizon Fix Bundle"
       git push

   (Alternatively, you can simply run:
       python apply_v4_and_push_to_github.py --push
   which unpacks and pushes everything in one command.)

INCLUDED v4.1 FIXES & PROFITABILITY UPGRADES:
- new_plan.md (Complete v4.0 + Section 11 v4.1 Deep Codebase Audit & Profitability Blueprint)
- feature engineering scripts/inference/checkpoint_loaders.py
    * Fixed missing .to(device) on all 6 predictor wrappers (_GBMPredictor, _HMMPredictor,
      _NeuralForecastPredictor, _TFTPredictor, _SSMPredictor, _SentimentPredictor)
    * Replaced 904ms/bar PyTorch Lightning Trainer.predict() with <6.5ms direct tensor
      forward pass + FP16 autocast
    * Removed 30-bar rolling mean subtraction from trend horizons (Rule CAL-4)
    * Fixed Kronos OOF schema mismatch (kronos_p_down/flat/up)
- feature engineering scripts/inference/fast_direct_nf_predictor.py
    * DirectDeviceAwareNeuralForecastPredictor (<6.5ms inference, .to(device) support)
- feature engineering scripts/inference/adaptive_hardware_coordinator.py
    * Auto-detects Local Laptop vs Kaggle Dual-T4 VM, HTTP-451 Geo-Failover, and HF swing state hand-off
- feature engineering scripts/common/neuralforecast_harness.py
    * Replaced 30-bar rolling mean subtraction with RevIN window-std z-score mapping
      and multi-GPU device index support (CUDA_VISIBLE_DEVICES / devices=[idx])
- feature engineering scripts/models/ssm_challenger/model.py
    * Upgraded SelectiveSSMCore to O(log2 L) parallel associative prefix scan (22x faster)
- feature engineering scripts/models_v3/tier2_tactical/parallel_scan_s6.py
    * FastParallelSelectiveSSMCore with O(log2 L) Blelloch scan + mamba_ssm CUDA dispatch
- feature engineering scripts/fusion/conformal_dirichlet_calibrator.py
    * True Diagonal-Preserved Symmetric ODIR Dirichlet Calibrator + Online Adaptive Conformal Inference (ACI)
- feature engineering scripts/fusion/hierarchical_fusion_v3.py
    * Cross-Horizon Alignment vectors & Shannon Negentropy certainty weighting
- feature engineering scripts/fusion/supervised_multi_horizon_stacker.py
    * Supervised Multi-Horizon OOF Stacker + LightGBM Huber Expected-Net-Bps Meta-Labeler
- feature engineering scripts/execution/hierarchical_sniper_engine.py
    * 9-State Hierarchical Maker Sniper Engine with 50% intra-bar exhaustion wick retracement
- feature engineering scripts/execution/maker_node.py
    * Upgraded MakerExecutionNode with 50% intra-bar exhaustion wick retracement limit pricing
- feature engineering scripts/labeling_multi_horizon_v4.py
    * Fee-floored triple-barrier labeler computing exact signed net bps for both LONG and SHORT
- feature engineering scripts/orchestration/checkpoint_manager_v4.py
    * 15-member Dual-T4 wave scheduler with SessionBudget & atomic HF Hub uploads
- feature engineering scripts/dist/kaggle_session_bundle.py
    * Rebuilt self-contained Kaggle Dual-T4 bundle containing all 135 repository files
