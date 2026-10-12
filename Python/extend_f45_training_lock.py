"""Create a new training lock from explicitly retired development rounds."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from select_f45_activation_threshold_v6 import load_manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--retired-round',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('new output required')
    source=json.loads(a.source.read_bytes())
    records=list(source['training']['captures'])
    hashes={r['sha256'] for r in records}
    added=[]
    for path in a.retired_round:
        for item in load_manifest(path):
            if item['sha256'] in hashes: continue
            records.append(item); added.append(item); hashes.add(item['sha256'])
    for item in records:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('source capture hash mismatch')
    result={'schema':'f45_expanded_training_lock_v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_lock_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
            'training':{'captures':records},'added_captures':added,
            'retired_rounds':[str(p.resolve()) for p in a.retired_round],
            'warning':'All included rounds are training data from now on; never blind validation.',
            'deployment_eligible':False,'physical_15hz_verified':False}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'training_count':len(records),'added':len(added)}))
