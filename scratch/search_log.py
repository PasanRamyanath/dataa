with open('scratch/live_paper_1h_updated.log', encoding='utf-8') as f:
    for line in f:
        if 'Closed' in line or 'Trade' in line or 'Bar 14' in line or 'Filled' in line or 'REVERSAL' in line or 'STOP' in line:
            print(line.strip())
