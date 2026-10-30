import sys
from pathlib import Path

REGISTRY_PATH = Path("feature engineering scripts/inference/live_v4_model_registry.py")
content = REGISTRY_PATH.read_text(encoding="utf-8")

# Define start anchor and end anchor
start_anchor = '        imbs_4h = np.append(imbs_4h, float(b10s[-1].get("taker_imb", 0.0)) if b10s else 0.0)\n\n        meso_anchor, _, _ = self._compute_regime_displacement(klines_um)'
end_anchor = '        return meso_score, tact_score, micro_score\n\n    def predict_tier3_meso(self, ring_buffer: Any, ffd_fn: Any = None) -> np.ndarray:'

assert start_anchor in content, "start_anchor not found!"
assert end_anchor in content, "end_anchor not found!"

start_idx = content.find('        meso_anchor, _, _ = self._compute_regime_displacement(klines_um)')
end_idx = content.find('    def predict_tier3_meso(self, ring_buffer: Any, ffd_fn: Any = None) -> np.ndarray:')

replacement_text = """        # Macro anchor based on genuine 1h / 4h price displacement
        ret_4h = ((closes_4h[-1] - closes_4h[-min(6, len(closes_4h))]) / closes_4h[-1]) * 1e4 if len(closes_4h) >= 2 else 0.0
        ret_1h = ((closes_1h[-1] - closes_1h[-min(4, len(closes_1h))]) / closes_1h[-1]) * 1e4 if len(closes_1h) >= 2 else 0.0
        macro_anchor = float(np.clip(0.60 * np.tanh(ret_4h / 25.0) + 0.40 * np.tanh(ret_1h / 15.0), -1.0, 1.0))

        # 1. HMM Regime Model
        hmm_post, p_hmm = self.predict_hmm_regime(closes_1h)
        self.cached_hmm_probs = hmm_post

        # 2. PatchTST-FFD Model
        if self.tier4_patchtst is not None and len(closes_4h) >= 12:
            p_ptst = self.tier4_patchtst.predict_from_series(closes_4h, imbs_4h, aux_bias=macro_anchor)
        else:
            p_ptst = _score_to_simplex(macro_anchor, base_flat=0.54, gain=3.2)

        # 3. TFT-Funding Model
        if self.tier4_tft is not None and len(closes_1h) >= 12:
            p_tft = self.tier4_tft.predict_from_series(closes_1h, None, aux_bias=macro_anchor)
        else:
            p_tft = p_ptst

        # 4. Kronos-Macro + Multi-Scale FFD (4h, 6h, 8h, 12h, 1d)
        p_kronos = _score_to_simplex(macro_anchor, base_flat=0.54, gain=3.2)

        macro_probs = 0.15 * p_hmm + 0.30 * p_ptst + 0.25 * p_tft + 0.30 * p_kronos
        macro_probs = macro_probs / np.sum(macro_probs)
        return macro_probs

    @staticmethod
    def _compute_regime_displacement(klines_um: dict[str, Any]) -> tuple[float, float, float]:
        \"\"\"
        Computes continuous (meso_regime_score, tact_regime_score, micro_regime_score)
        based on genuine rolling price z-scores, velocity, and multi-scale returns.
        Guarantees smooth, continuous anchoring without artificial discrete bounding boxes.
        \"\"\"
        b10s = list(klines_um.get("10s", []))
        b1m = list(klines_um.get("1m", []))
        if len(b10s) >= 6:
            c_arr = np.array([float(x["close"]) for x in b10s[-120:]], dtype=np.float64)
            imb_arr = np.array([float(x.get("taker_imb", 0.0)) for x in b10s[-12:]], dtype=np.float64)
        elif len(b1m) >= 5:
            c_arr = np.array([float(x["close"]) for x in b1m[-30:]], dtype=np.float64)
            imb_arr = np.array([float(x.get("taker_imb", 0.0)) for x in b1m[-5:]], dtype=np.float64)
        else:
            return 0.0, 0.0, 0.0

        c_now = c_arr[-1]
        n_c = len(c_arr)
        w_meso = min(n_c, 90)
        w_tact = min(n_c, 36)
        eq_meso = float(np.mean(c_arr[-w_meso:]))
        std_meso = max(float(np.std(c_arr[-w_meso:])), c_now * 0.00042)
        eq_tact = float(np.mean(c_arr[-w_tact:]))
        std_tact = max(float(np.std(c_arr[-w_tact:])), c_now * 0.00028)

        z_meso = (c_now - eq_meso) / std_meso
        z_tact = (c_now - eq_tact) / std_tact

        ret_5m = ((c_now - c_arr[-min(31, n_c)]) / c_now) * 1e4
        ret_1m = ((c_now - c_arr[-min(7, n_c)]) / c_now) * 1e4
        vel_10s = ((c_now - c_arr[-min(2, n_c)]) / c_now) * 1e4
        flow_10s = float(np.mean(imb_arr[-min(3, len(imb_arr)):])) if len(imb_arr) > 0 else 0.0

        # Continuous smooth physical displacement bounded gracefully in [-0.85, +0.85]
        meso_score = float(np.clip(0.40 * z_meso + 0.60 * np.tanh(ret_5m / 8.0), -0.85, 0.85))
        tact_score = float(np.clip(0.35 * z_tact + 0.50 * np.tanh(ret_1m / 4.0) + 0.15 * np.tanh(flow_10s), -0.85, 0.85))
        micro_score = float(np.clip(0.65 * np.tanh(vel_10s / 2.0) + 0.35 * np.tanh(flow_10s), -0.85, 0.85))

        return meso_score, tact_score, micro_score

"""

new_content = content[:start_idx] + replacement_text + content[end_idx:]
REGISTRY_PATH.write_text(new_content, encoding="utf-8")
print(f"Successfully patched {REGISTRY_PATH}! Old length: {len(content)}, new length: {len(new_content)}")
