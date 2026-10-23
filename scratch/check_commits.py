import os
import sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get('HF_TOKEN'))
repos = [
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-tier4_macro',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-tier3_meso',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-tier2_tactical',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-tier1_micro',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-fusion',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-gbm-family',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-hmm-regime',
    'p-j-r-1-2-3/Project-IMPJR-v4-Models-ssm-challenger',
    'p-j-r-1-2-3/Project-IMPJR-Models-gbm-family',
    'p-j-r-1-2-3/Project-IMPJR-Models-ssm-challenger'
]

for r in repos:
    try:
        commits = api.list_repo_commits(repo_id=r, repo_type='model')
        latest = commits[0]
        print(f'{r:50s}: {latest.created_at} | {latest.title}')
    except Exception as e:
        print(f'{r:50s}: ERROR {e}')
