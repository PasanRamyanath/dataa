with open('changelog.md', 'r', encoding='utf-8') as f:
    for line in f:
        if 'Cycle 59' in line and '|' in line:
            print(repr(line))
            break
