import os, json, subprocess, shutil, pathlib

for name, kf in [('pjramyanath', 'feature engineering scripts/kaggle_4.json'), ('impjr123', 'feature engineering scripts/kaggle_5.json'), ('pasanramyanath', 'feature engineering scripts/kaggle_6.json')]:
    stage = pathlib.Path('scratch') / f'quota_{name}'
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy(kf, stage / 'kaggle.json')
    meta = {
        'id': f'{name}/test-quota-check',
        'title': 'test-quota-check',
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
        f.write('print("Hello GPU")\n')
    env = os.environ.copy()
    env['KAGGLE_CONFIG_DIR'] = str(stage.resolve())
    res = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(stage.resolve())], env=env, capture_output=True, text=True)
    print(name, 'rc:', res.returncode, 'stdout:', res.stdout.strip(), 'stderr:', res.stderr.strip())
