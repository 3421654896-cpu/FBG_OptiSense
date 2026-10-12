"""Lock a simple CH1-only pilot before acquiring independent validation contacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from audit_candidate27_contact import load_contact
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file, now_text


def train(paths, output):
    if output.exists(): raise FileExistsError(output)
    records=[load_contact(p)[0] for p in paths]
    if len({r['point_index'] for r in records})!=len(records) or len(records)<2:
        raise ValueError('One independent prototype per physical point required')
    if len({(r['route_crc32'],r['first_delay_us'],r['spacing_us'],r['depth_mm']) for r in records})!=1:
        raise ValueError('Training route/timing/depth mismatch')
    prototypes=[dict(point=r['point_index'],source=r['source'],sha256=r['sha256'],
                     vector=r['samples'][1]['shape_delta']) for r in records]
    record=records[0]
    report=dict(schema='candidate27_position_pilot_v1',created=now_text(),deployed=False,
        route_crc32=record['route_crc32'],first_delay_us=record['first_delay_us'],
        spacing_us=record['spacing_us'],depth_mm=record['depth_mm'],prototypes=prototypes,
        method='Second physical read; per-FBG unit-vector shape minus declared no-contact baseline; unweighted Euclidean nearest prototype; no temporal smoothing.',
        physical_15hz_verified=False,full_spatial_accuracy_verified=False,
        scope='Sparse fixed-position experiment only, not a full-silicone position model.')
    save_checkpoint(output,report)
    print('locked',sha256_file(output),'points',[r['point'] for r in prototypes])


def validate(model_path, paths, output):
    if output.exists(): raise FileExistsError(output)
    model=json.loads(model_path.read_text('utf-8'))
    prototypes=np.array([p['vector'] for p in model['prototypes']],dtype=float)
    train_hashes={p['sha256'] for p in model['prototypes']}
    results=[]
    for path in paths:
        row=load_contact(path)[0]
        if row['sha256'] in train_hashes:
            raise ValueError('Training capture cannot be scored as independent validation')
        if any(row[k]!=model[k] for k in ('route_crc32','first_delay_us','spacing_us','depth_mm')):
            raise ValueError('Validation route/timing/depth mismatch')
        # Point labels are not inputs to these features or distance calculations.
        features=np.asarray(row['samples'][1]['held_shape_features'],dtype=float)
        distances=np.sqrt(np.mean((features[:,None]-prototypes[None,:])**2,axis=2))
        predictions=[model['prototypes'][i]['point'] for i in np.argmin(distances,axis=1)]
        median_feature=np.array(row['samples'][1]['shape_delta'])
        trial_distances=np.sqrt(np.mean((prototypes-median_feature)**2,axis=1))
        results.append(dict(source=row['source'],sha256=row['sha256'],truth=row['point_index'],
            held_frames=len(features),correct_frames=predictions.count(row['point_index']),
            frame_confusion=dict(Counter(predictions)),trial_prediction=model['prototypes'][int(np.argmin(trial_distances))]['point'],
            distances_to_prototypes=trial_distances.tolist()))
    report=dict(schema='candidate27_position_pilot_validation_v1',model_sha256=sha256_file(model_path),
        validation=results,deployed=False,physical_15hz_verified=False,full_spatial_accuracy_verified=False,
        scope='Separate physical contacts, but successive held frames are correlated; not continuous-space accuracy or frequency validation.')
    save_checkpoint(output,report)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','validate'));p.add_argument('captures',type=Path,nargs='+')
    p.add_argument('--model',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='train': train(args.captures,args.output)
    else:
        if args.model is None: p.error('--model required')
        validate(args.model,args.captures,args.output)
