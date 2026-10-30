import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')

for fpath in ['scratch/live_paper_1h_updated.log', 'scratch/live_champion_challenger.log', 'scratch/champion_challenger_final_results.json']:
    if os.path.exists(fpath):
        print(f"=== {fpath} ===")
        with open(fpath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'Challenger_2' in content:
                print("Found Challenger_2 occurrences:", content.count('Challenger_2'))
