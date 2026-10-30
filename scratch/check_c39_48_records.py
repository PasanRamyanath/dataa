import json, glob, os

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
c39_48 = [f"archive_cycle{i}" for i in range(39, 49)]

for cyc in c39_48:
    path = os.path.join(base_dir, cyc)
    results_f = os.path.join(path, "champion_challenger_final_results.json")
    if os.path.exists(results_f):
        with open(results_f, "r") as f:
            data = json.load(f)
            winner = data.get("tournament_winner")
            pnl = data.get("winner_net_pnl_bps")
            wr = data.get("winner_win_rate_pct")
            trades = data.get("winner_trades")
            print(f"{cyc}: Winner={winner:<12} Net={pnl:>+6.2f}b WR={wr:>5.1f}% Trades={trades}")
    else:
        print(f"{cyc}: No results json")
