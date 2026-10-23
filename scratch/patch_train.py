path = 'feature engineering scripts/models_v3/train_all_multi_horizon.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add stacker.save(fusion_dir)
old1 = "        calibrator = SymmetricODIRDirichletCalibrator()\n        stacker = SupervisedMultiHorizonOOFStacker()\n        \n        metrics_path = fusion_dir / \"metrics.json\""
new1 = "        calibrator = SymmetricODIRDirichletCalibrator()\n        stacker = SupervisedMultiHorizonOOFStacker()\n        stacker.save(fusion_dir)\n        \n        metrics_path = fusion_dir / \"metrics.json\""

# 2. Pass force_retrain to execute_dual_t4_tier_training_wave
old2 = "        ok = execute_dual_t4_tier_training_wave(\n            tier_name=tier,\n            member_runners=runners,\n            budget=budget,\n            output_root=output_root,\n            model_repo_id=target_repo,\n            hf_token=hf_token,\n        )"
new2 = "        ok = execute_dual_t4_tier_training_wave(\n            tier_name=tier,\n            member_runners=runners,\n            budget=budget,\n            output_root=output_root,\n            model_repo_id=target_repo,\n            hf_token=hf_token,\n            force_retrain=force_retrain,\n        )"

# Normalise newlines for replacement
text_norm = text.replace("\r\n", "\n")
assert old1 in text_norm, "old1 not found"
assert old2 in text_norm, "old2 not found"

text_norm = text_norm.replace(old1, new1).replace(old2, new2)
with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(text_norm)
print("Successfully patched train_all_multi_horizon.py")
