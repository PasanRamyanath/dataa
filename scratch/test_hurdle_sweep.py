import sys

from run_full_unified_peak_replay import UnifiedPeakExecutionEngineReplay, archives

for shield_h, shield_fl in [(3.8, 4.15), (4.5, 4.5), (4.5, 5.25), (5.0, 5.0)]:
    engine = UnifiedPeakExecutionEngineReplay()
    engine.shield_mfe = shield_h
    engine.shield_floor = shield_fl
    all_res = []
    for arch in archives:
        r = engine.replay_archive(arch)
        if r and r['trades'] > 0:
            all_res.append(r)
    tot_tr = sum(r['trades'] for r in all_res)
    tot_w = sum(r['wins'] for r in all_res)
    tot_net = sum(r['net_pnl'] for r in all_res)
    wr = tot_w / tot_tr * 100.0 if tot_tr > 0 else 0.0
    avg = tot_net / tot_tr if tot_tr > 0 else 0.0
    w100 = len([r for r in all_res if r['win_rate'] == 100.0])
    prof = len([r for r in all_res if r['net_pnl'] > 0.0])
    print(f"Hurdle: {shield_h:.1f}b / Floor: {shield_fl:.2f}b -> Tr: {tot_tr:<3} | WR: {wr:>5.1f}% | Net: {tot_net:>+8.2f} bps | Avg: {avg:>+5.2f} bps | 100% WR: {w100}/{len(all_res)} | Profitable: {prof}/{len(all_res)}")
