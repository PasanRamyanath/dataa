import re
from pathlib import Path

path = Path("feature engineering scripts/streaming/run_live_hierarchical_session.py")
content = path.read_text(encoding="utf-8")

# 1. Update self.stats
pattern1 = 'tf: {"total": 0, "correct_dir": 0, "active_total": 0, "active_correct": 0}'
replace1 = 'tf: {"total": 0, "correct_dir": 0, "active_total": 0, "active_correct": 0, "sign_total": 0, "sign_correct": 0}'
assert pattern1 in content, "pattern1 not found"
content = content.replace(pattern1, replace1, 1)

# 2. Update active accuracy & sign accuracy
pattern2 = """                    if tick["primary_dir"] in (0, 2):
                        st["active_total"] += 1
                        if tick["primary_dir"] == actual_dir:
                            st["active_correct"] += 1"""
replace2 = """                    if tick["primary_dir"] in (0, 2):
                        st["active_total"] += 1
                        if tick["primary_dir"] == actual_dir:
                            st["active_correct"] += 1
                        if ret_bps != 0.0:
                            actual_sign = 2 if ret_bps > 0 else 0
                            st["sign_total"] += 1
                            if tick["primary_dir"] == actual_sign:
                                st["sign_correct"] += 1"""
# Normalize line endings for replacement
content_norm = content.replace("\r\n", "\n")
pattern2_norm = pattern2.replace("\r\n", "\n")
replace2_norm = replace2.replace("\r\n", "\n")
assert pattern2_norm in content_norm, "pattern2 not found"
content_norm = content_norm.replace(pattern2_norm, replace2_norm, 1)

# 3. Update get_summary
pattern3_norm = """            res[tf] = {
                "evaluated_samples": tot,
                "overall_accuracy_pct": round(acc, 2),
                "active_signals_count": act_tot,
                "active_precision_pct": round(act_prec, 2),
            }"""
replace3_norm = """            sign_tot = st.get("sign_total", 0)
            sign_acc = (st.get("sign_correct", 0) / sign_tot * 100.0) if sign_tot > 0 else 0.0
            res[tf] = {
                "evaluated_samples": tot,
                "overall_accuracy_pct": round(acc, 2),
                "active_signals_count": act_tot,
                "active_precision_pct": round(act_prec, 2),
                "sign_accuracy_pct": round(sign_acc, 2),
                "sign_samples": sign_tot,
            }"""
assert pattern3_norm in content_norm, "pattern3 not found"
content_norm = content_norm.replace(pattern3_norm, replace3_norm, 1)

# 4. Include 15m in periodic print
pattern4_norm = 'for tf in ["10s", "30s", "1m", "5m"]:'
replace4_norm = 'for tf in ["10s", "30s", "1m", "5m", "15m"]:'
assert pattern4_norm in content_norm, "pattern4 not found"
content_norm = content_norm.replace(pattern4_norm, replace4_norm, 1)

path.write_text(content_norm, encoding="utf-8")
print("Enhanced forward accuracy telemetry applied successfully!")
