"""Optional operational sidecars, excluded from scientific artifact identity."""
import json
import os
from pathlib import Path
import time


def event(stage, seconds, **fields):
    root=os.environ.get('AUTOFE_PROFILE_DIR')
    if not root:return
    directory=Path(root); directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'operational_{os.getpid()}.jsonl').open('a',encoding='utf-8') as output:
        output.write(json.dumps({'utc_epoch':time.time(),'pid':os.getpid(),'stage':stage,'seconds':seconds,**fields})+'\n')
