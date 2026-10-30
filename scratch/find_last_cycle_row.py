import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('changelog.md', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip().startswith('| **Cycle') and '|' in line:
            last_line = line

print("Last cycle row in table:")
print(last_line)
