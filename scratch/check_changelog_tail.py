with open('changelog.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in changelog.md: {len(lines)}")
print("Last 25 lines:")
for l in lines[-25:]:
    print(l, end='')
