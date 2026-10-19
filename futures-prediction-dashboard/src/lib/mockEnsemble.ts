import { EnsemblePrediction, MemberPrediction } from "@/types";

const MEMBERS = [
  { name: "patchtst", baseWeight: 0.22, dirAcc: 0.5262, latencyRange: [25, 45] },
  { name: "tft", baseWeight: 0.18, dirAcc: 0.5185, latencyRange: [40, 75] },
  { name: "gbm-family", baseWeight: 0.20, dirAcc: 0.5175, latencyRange: [3, 8] },
  { name: "nhits", baseWeight: 0.14, dirAcc: 0.5109, latencyRange: [15, 30] },
  { name: "kronos-finetune", baseWeight: 0.12, dirAcc: 0.5020, latencyRange: [50, 95] },
  { name: "sentiment-branch", baseWeight: 0.05, dirAcc: 0.5008, latencyRange: [2, 5] },
  { name: "ssm-challenger", baseWeight: 0.05, dirAcc: 0.4963, latencyRange: [18, 35] },
  { name: "itransformer", baseWeight: 0.04, dirAcc: 0.4962, latencyRange: [20, 40] },
];

const REGIMES: Array<{ state: 0 | 1 | 2; name: string; multiplier: number }> = [
  { state: 0, name: "Low Volatility / Mean-Reversion", multiplier: 1.2 },
  { state: 1, name: "Directional Momentum Trend", multiplier: 1.0 },
  { state: 2, name: "High Volatility / Liquidation Cascade", multiplier: 0.4 },
];

let tickCounter = 0;
let currentRegimeIndex = 0;

export function generateMockEnsemblePrediction(
  currentPrice: number,
  recentPriceChangePct: number = 0,
  metaThreshold: number = 0.60
): EnsemblePrediction {
  tickCounter++;

  // Switch regime every ~18 ticks (3 minutes)
  if (tickCounter % 18 === 0) {
    currentRegimeIndex = (currentRegimeIndex + 1) % REGIMES.length;
  }
  const currentRegime = REGIMES[currentRegimeIndex];

  // Base market drift momentum
  const momentumBias = Math.max(Math.min(recentPriceChangePct * 10, 0.4), -0.4);

  const members: Record<string, MemberPrediction> = {};
  let weightedProbDown = 0;
  let weightedProbFlat = 0;
  let weightedProbUp = 0;
  let totalWeight = 0;

  for (const m of MEMBERS) {
    // Generate soft probability simplex
    const noiseDown = (Math.random() - 0.5) * 0.12 - momentumBias * 0.5;
    const noiseUp = (Math.random() - 0.5) * 0.12 + momentumBias * 0.5;
    const rawDown = Math.max(0.15, 0.33 + noiseDown);
    const rawUp = Math.max(0.15, 0.33 + noiseUp);
    const rawFlat = Math.max(0.15, 1.0 - rawDown - rawUp);

    const sum = rawDown + rawFlat + rawUp;
    const prob_down = parseFloat((rawDown / sum).toFixed(4));
    const prob_flat = parseFloat((rawFlat / sum).toFixed(4));
    const prob_up = parseFloat((1.0 - prob_down - prob_flat).toFixed(4));

    const dir: -1 | 0 | 1 = prob_up > prob_down && prob_up > prob_flat ? 1 : prob_down > prob_up && prob_down > prob_flat ? -1 : 0;
    const predRet = (prob_up - prob_down) * 0.0035;
    const latency = Math.floor(Math.random() * (m.latencyRange[1] - m.latencyRange[0])) + m.latencyRange[0];

    members[m.name] = {
      name: m.name,
      prob_down,
      prob_flat,
      prob_up,
      primary_direction: dir,
      predicted_return: parseFloat(predRet.toFixed(6)),
      latency_ms: latency,
      weight: m.baseWeight,
    };

    weightedProbDown += prob_down * m.baseWeight;
    weightedProbFlat += prob_flat * m.baseWeight;
    weightedProbUp += prob_up * m.baseWeight;
    totalWeight += m.baseWeight;
  }

  // Raw Level-2 Stacking probabilities
  const sDown = weightedProbDown / totalWeight;
  const sFlat = weightedProbFlat / totalWeight;
  const sUp = weightedProbUp / totalWeight;
  const rawSum = sDown + sFlat + sUp;
  const raw_proba: [number, number, number] = [
    parseFloat((sDown / rawSum).toFixed(4)),
    parseFloat((sFlat / rawSum).toFixed(4)),
    parseFloat((sUp / rawSum).toFixed(4)),
  ];

  // Dirichlet ODIR Calibrator (sharpens confidence without distortion)
  const dir_down = Math.pow(raw_proba[0], 1.15);
  const dir_flat = Math.pow(raw_proba[1], 0.95);
  const dir_up = Math.pow(raw_proba[2], 1.15);
  const dirSum = dir_down + dir_flat + dir_up;
  const calibrated_proba: [number, number, number] = [
    parseFloat((dir_down / dirSum).toFixed(4)),
    parseFloat((dir_flat / dirSum).toFixed(4)),
    parseFloat((dir_up / dirSum).toFixed(4)),
  ];

  // Direction & Confidence
  const sorted = [...calibrated_proba].sort((a, b) => a - b);
  const confidence = parseFloat((sorted[2] - sorted[1]).toFixed(4));
  const primary_direction: -1 | 0 | 1 =
    calibrated_proba[2] > calibrated_proba[0] && calibrated_proba[2] > calibrated_proba[1]
      ? 1
      : calibrated_proba[0] > calibrated_proba[2] && calibrated_proba[0] > calibrated_proba[1]
      ? -1
      : 0;

  // Secondary Meta-Labeling Filter Output: P(Act)
  // Higher confidence and lower flat mass yields higher act probability
  const baseAct = 0.42 + confidence * 1.5 - calibrated_proba[1] * 0.3;
  // Inject occasional high conviction signals (~20% of ticks exceed tau = 0.60)
  const burst = Math.random() < 0.22 ? 0.22 + Math.random() * 0.14 : 0.0;
  const act_probability = parseFloat(Math.min(Math.max(baseAct + burst, 0.32), 0.88).toFixed(4));

  const is_trade_allowed = act_probability >= metaThreshold && primary_direction !== 0;

  // Kelly target fraction
  let target_fraction = 0.0;
  if (is_trade_allowed) {
    const betMultiplier = Math.min(Math.max((act_probability - metaThreshold) / (1.0 - metaThreshold), 0.1), 1.0);
    const rawSizing = primary_direction * Math.pow(confidence, 2) * betMultiplier * currentRegime.multiplier;
    target_fraction = parseFloat(Math.max(Math.min(rawSizing, 0.25), -0.25).toFixed(4));
  }

  const bet_size = parseFloat((act_probability * confidence).toFixed(4));

  return {
    timestamp: new Date().toISOString(),
    bar_index: tickCounter,
    symbol: "BTCUSDT",
    members,
    regime_state: currentRegime.state,
    regime_name: currentRegime.name,
    regime_multiplier: currentRegime.multiplier,
    raw_proba,
    calibrated_proba,
    primary_direction,
    confidence,
    act_probability,
    bet_size,
    is_trade_allowed,
    meta_threshold: metaThreshold,
    target_fraction,
    fusion_stage: "stacking+dirichlet_odir+meta_labeling",
    latency_seconds: 0.142,
  };
}
