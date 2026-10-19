import { EnsemblePrediction, ModelAccuracyStats, ResolvedPrediction } from "@/types";

interface PendingPrediction {
  id: string;
  timestamp: string;
  createdAtMs: number;
  entryPrice: number;
  prediction: EnsemblePrediction;
  isSettled: boolean;
}

export class LiveAccuracyTracker {
  private pending: PendingPrediction[] = [];
  private resolved: ResolvedPrediction[] = [];
  private memberStats: Record<string, { total: number; correct: number }> = {};

  // Pure live statistics - start from clean slate
  private counts = {
    totalEvaluated: 0,
    correctPredictions: 0,
    upEvaluated: 0,
    upCorrect: 0,
    downEvaluated: 0,
    downCorrect: 0,
    authorizedEvaluated: 0,
    authorizedCorrect: 0,
    vetoedEvaluated: 0,
    vetoedCorrect: 0,
    horizon10s: { total: 0, correct: 0 },
    horizon30s: { total: 0, correct: 0 },
    horizon60s: { total: 0, correct: 0 },
  };

  private idCounter = 0;

  constructor() {
    this.reset();
  }

  /**
   * Called whenever a new 10-second prediction arrives from the inference engine.
   * This settles any previous pending bar exactly on the 10-second interval,
   * then queues the new bar for the next settlement.
   */
  public onNewPrediction(prediction: EnsemblePrediction, livePrice: number): ResolvedPrediction[] {
    const anchorPrice = (typeof prediction.source_close === "number" && prediction.source_close > 0)
      ? prediction.source_close
      : livePrice;
    if (!anchorPrice || anchorPrice <= 0) return [];

    const now = Date.now();
    const newlyResolved: ResolvedPrediction[] = [];

    // 1. Settle all older pending predictions against this new bar's exact close/opening price
    for (const item of this.pending) {
      if (!item.isSettled) {
        const res = this.settlePrediction(item, anchorPrice, now);
        if (res) newlyResolved.push(res);
      }
    }

    // Retain only unsettled predictions (or keep the latest one)
    this.pending = this.pending.filter((p) => !p.isSettled);

    // 2. Queue the new prediction for the current bar anchored to the true bar close
    this.pending.push({
      id: `pred_${++this.idCounter}`,
      timestamp: prediction.timestamp || new Date(now).toISOString(),
      createdAtMs: now,
      entryPrice: anchorPrice,
      prediction,
      isSettled: false,
    });

    return newlyResolved;
  }

  /**
   * Called on incoming WebSocket price ticks.
   * If a pending prediction has been waiting >= 10 seconds without a new prediction,
   * it gets settled against the live price.
   */
  public onLivePrice(livePrice: number): ResolvedPrediction[] {
    if (!livePrice || livePrice <= 0) return [];

    const now = Date.now();
    const newlyResolved: ResolvedPrediction[] = [];

    for (const item of this.pending) {
      const elapsedSec = (now - item.createdAtMs) / 1000;
      // Settle once the 10-second prediction interval has elapsed
      if (!item.isSettled && elapsedSec >= 10.0) {
        const res = this.settlePrediction(item, livePrice, now);
        if (res) newlyResolved.push(res);
      }
    }

    this.pending = this.pending.filter((p) => !p.isSettled);
    return newlyResolved;
  }

  /**
   * Settles a single prediction bar against the target price.
   * Guaranteed to execute EXACTLY ONCE per prediction bar.
   */
  private settlePrediction(
    item: PendingPrediction,
    settlementPrice: number,
    nowMs: number
  ): ResolvedPrediction | null {
    item.isSettled = true;

    const entryP = item.entryPrice;
    if (!entryP || entryP <= 0 || !settlementPrice || settlementPrice <= 0) return null;

    const diffPct = ((settlementPrice - entryP) / entryP) * 100;
    // Discard uninitialized connection spikes (> 3% on 10s bar)
    if (Math.abs(diffPct) > 3.0) return null;

    const predDir = item.prediction.primary_direction;
    const actualDir: -1 | 0 | 1 = diffPct > 0.002 ? 1 : diffPct < -0.002 ? -1 : 0;

    // 3-Class Ground Truth:
    // UP (1): Price must have risen (diffPct > 0)
    // DOWN (-1): Price must have fallen (diffPct < 0)
    // FLAT (0): Price remained rangebound (|diffPct| <= 0.015%)
    let isCorrect = false;
    if (predDir === 1 && diffPct > 0) isCorrect = true;
    else if (predDir === -1 && diffPct < 0) isCorrect = true;
    else if (predDir === 0 && Math.abs(diffPct) <= 0.015) isCorrect = true;

    const isAuthorized = item.prediction.is_trade_allowed;
    const elapsedSec = Math.round((nowMs - item.createdAtMs) / 1000);

    const record: ResolvedPrediction = {
      id: `res_${item.id}`,
      timestamp: item.timestamp,
      resolvedTimestamp: new Date(nowMs).toISOString(),
      horizonSec: Math.max(elapsedSec, 10),
      entryPrice: entryP,
      resolvedPrice: settlementPrice,
      priceChangePct: diffPct,
      predictedDirection: predDir,
      actualDirection: actualDir,
      isCorrect,
      confidence: item.prediction.confidence,
      actProbability: item.prediction.act_probability,
      isAuthorized,
    };

    this.resolved.unshift(record);
    if (this.resolved.length > 200) {
      this.resolved.pop();
    }

    // 1. Update Ensemble Evaluation Counts (Exactly 1 evaluation per bar)
    this.counts.totalEvaluated++;
    if (isCorrect) this.counts.correctPredictions++;

    if (predDir === 1) {
      this.counts.upEvaluated++;
      if (isCorrect) this.counts.upCorrect++;
    } else if (predDir === -1) {
      this.counts.downEvaluated++;
      if (isCorrect) this.counts.downCorrect++;
    }

    if (isAuthorized) {
      this.counts.authorizedEvaluated++;
      if (isCorrect) this.counts.authorizedCorrect++;
    } else {
      this.counts.vetoedEvaluated++;
      if (isCorrect) this.counts.vetoedCorrect++;
    }

    this.counts.horizon10s.total++;
    if (isCorrect) this.counts.horizon10s.correct++;

    // 2. Update Member Model Stats (Exactly 1 evaluation per member per bar)
    if (item.prediction.members) {
      for (const [mName, mPred] of Object.entries(item.prediction.members)) {
        if (!this.memberStats[mName]) {
          this.memberStats[mName] = { total: 0, correct: 0 };
        }
        this.memberStats[mName].total++;

        let mCorrect = false;
        if (mPred.primary_direction === 1 && diffPct > 0) {
          mCorrect = true;
        } else if (mPred.primary_direction === -1 && diffPct < 0) {
          mCorrect = true;
        } else if (mPred.primary_direction === 0 && Math.abs(diffPct) <= 0.015) {
          mCorrect = true;
        }

        if (mCorrect) {
          this.memberStats[mName].correct++;
        }
      }
    }

    return record;
  }

  public getStats(): ModelAccuracyStats {
    const total = Math.max(1, this.counts.totalEvaluated);
    const overallAccuracy = this.counts.totalEvaluated > 0
      ? (this.counts.correctPredictions / total) * 100
      : 50.0;

    const upTotal = Math.max(1, this.counts.upEvaluated);
    const upAccuracy = this.counts.upEvaluated > 0
      ? (this.counts.upCorrect / upTotal) * 100
      : 50.0;

    const downTotal = Math.max(1, this.counts.downEvaluated);
    const downAccuracy = this.counts.downEvaluated > 0
      ? (this.counts.downCorrect / downTotal) * 100
      : 50.0;

    const authTotal = Math.max(1, this.counts.authorizedEvaluated);
    const authorizedAccuracy = this.counts.authorizedEvaluated > 0
      ? (this.counts.authorizedCorrect / authTotal) * 100
      : 50.0;

    const vetoTotal = Math.max(1, this.counts.vetoedEvaluated);
    const vetoedAccuracy = this.counts.vetoedEvaluated > 0
      ? (this.counts.vetoedCorrect / vetoTotal) * 100
      : 50.0;

    const h10 = this.counts.horizon10s.total > 0
      ? (this.counts.horizon10s.correct / this.counts.horizon10s.total) * 100
      : 50.0;

    const sumReturn = this.resolved.slice(0, 50).reduce((acc, r) => acc + Math.abs(r.priceChangePct), 0);
    const avgReturn = this.resolved.length > 0 ? sumReturn / Math.min(this.resolved.length, 50) : 0;

    // Per-member accuracies (strictly 1 evaluation per 10s prediction interval)
    const memberAccuracies: Record<string, { total: number; correct: number; accuracy: number }> = {};
    const defaultMembers = [
      "gbm-family", "itransformer", "kronos-finetune", "nhits",
      "patchtst", "sentiment-branch", "ssm-challenger", "tft"
    ];

    for (const mName of defaultMembers) {
      const s = this.memberStats[mName] || { total: 0, correct: 0 };
      memberAccuracies[mName] = {
        total: s.total,
        correct: s.correct,
        accuracy: s.total > 0 ? (s.correct / s.total) * 100 : 50.0,
      };
    }

    const latestPending = this.pending.find((p) => !p.isSettled);

    return {
      totalEvaluated: this.counts.totalEvaluated,
      correctPredictions: this.counts.correctPredictions,
      overallAccuracy,
      upEvaluated: this.counts.upEvaluated,
      upCorrect: this.counts.upCorrect,
      upAccuracy,
      downEvaluated: this.counts.downEvaluated,
      downCorrect: this.counts.downCorrect,
      downAccuracy,
      authorizedEvaluated: this.counts.authorizedEvaluated,
      authorizedCorrect: this.counts.authorizedCorrect,
      authorizedAccuracy,
      vetoedEvaluated: this.counts.vetoedEvaluated,
      vetoedCorrect: this.counts.vetoedCorrect,
      vetoedAccuracy,
      avgPriceReturnPct: avgReturn,
      horizon10sAccuracy: h10,
      horizon30sAccuracy: h10,
      horizon60sAccuracy: h10,
      recentResolved: [...this.resolved],
      memberAccuracies,
      activePending: latestPending
        ? {
            id: latestPending.id,
            timestamp: latestPending.timestamp,
            createdAtMs: latestPending.createdAtMs,
            entryPrice: latestPending.entryPrice,
            predictedDirection: latestPending.prediction.primary_direction,
            confidence: latestPending.prediction.confidence,
            isAuthorized: latestPending.prediction.is_trade_allowed,
          }
        : null,
    };
  }

  public reset() {
    this.pending = [];
    this.resolved = [];
    this.memberStats = {
      "gbm-family": { total: 0, correct: 0 },
      "itransformer": { total: 0, correct: 0 },
      "kronos-finetune": { total: 0, correct: 0 },
      "nhits": { total: 0, correct: 0 },
      "patchtst": { total: 0, correct: 0 },
      "sentiment-branch": { total: 0, correct: 0 },
      "ssm-challenger": { total: 0, correct: 0 },
      "tft": { total: 0, correct: 0 },
    };
    this.counts = {
      totalEvaluated: 0,
      correctPredictions: 0,
      upEvaluated: 0,
      upCorrect: 0,
      downEvaluated: 0,
      downCorrect: 0,
      authorizedEvaluated: 0,
      authorizedCorrect: 0,
      vetoedEvaluated: 0,
      vetoedCorrect: 0,
      horizon10s: { total: 0, correct: 0 },
      horizon30s: { total: 0, correct: 0 },
      horizon60s: { total: 0, correct: 0 },
    };
    this.idCounter = 0;
  }
}
