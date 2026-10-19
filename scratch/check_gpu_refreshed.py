import os, subprocess, json, pathlib

for name, kf in [('malaysiatrip', 'feature engineering scripts/kaggle_3.json'), ('rimramyanath', 'feature engineering scripts/kaggle_2.json'), ('pjramyanath', 'feature engineering scripts/kaggle_4.json')]:
    stage = pathlib.Path('scratch') / f'test_gpu_{name}'
    stage.mkdir(parents=True, exist_ok=True)
    meta = {
        'id': f'{name}/test-gpu-check-{name}',
        'title': f'test-gpu-check-{name}',
        'code_file': 'test.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
    }
    with open(stage / 'kernel-metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    with open(stage / 'test.py', 'w') as f:
        f.write('import torch\nprint("CUDA:", torch.cuda.is_available())\n')
    with open(kf) as f:
        c = json.load(f)
    env = os.environ.copy()
    env['KAGGLE_USERNAME'] = c['username']
    env['KAGGLE_KEY'] = c['key']
    res = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(stage.resolve())], env=env, capture_output=True, text=True)
    print(name, 'rc:', res.returncode, 'stdout:', res.stdout.strip(), 'stderr:', res.stderr.strip())
