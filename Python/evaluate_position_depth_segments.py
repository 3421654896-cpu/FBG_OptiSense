"""Exploratory joint position/depth inference from fixed spectral segments.

Two measured depths define each position's segment. No machine coordinates
enter inference. Intermediate-depth linearity is a hypothesis under test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_current_three_positions import capture_parts,spectral_feature
from displacement_pressure_reference import from_machine


def infer(query, segments):
    query=np.asarray(query,dtype=float)
    if query.shape!=(45,) or not np.isfinite(query).all(): raise ValueError('Invalid spectrum')
    candidates=[]
    for point,lo,hi,depth_lo,depth_hi in segments:
        lo,hi=np.asarray(lo),np.asarray(hi)
        direction=hi-lo
        norm=float(direction@direction)
        if norm<=1e-12: raise ValueError('Degenerate depth segment')
        t=float(np.clip((query-lo)@direction/norm,0,1))
        residual=float(np.sqrt(np.mean((query-(lo+t*direction))**2)))
        candidates.append(dict(point=int(point),estimated_indentation_mm=depth_lo+t*(depth_hi-depth_lo),residual=residual))
    return min(candidates,key=lambda r:r['residual'])


def read_label(path):
    data=json.loads(path.read_text('utf-8'))
    c=data['commissioning']
    status=next(e['status'] for e in c['events'] if e['kind']=='contact_reached_commanded_depth')
    return int(c['target']['point_index']),from_machine(c['target'],status)['indentation_mm']


def evaluate(manifest, query_path):
    spec=json.loads(manifest.read_text('utf-8'))
    paths=[(manifest.parent/p).resolve() for p in spec['training']]
    if query_path.resolve() in paths: raise ValueError('Query overlaps training')
    grouped={}
    for path in paths:
        base,held=capture_parts(path)
        point,depth=read_label(path)
        grouped.setdefault(point,[]).append((depth,spectral_feature(base,np.median(held,axis=0),'shape_delta')))
    segments=[]
    for point,entries in grouped.items():
        if len(entries)!=2: raise ValueError('Exactly two training depths per point required')
        entries.sort(key=lambda r:r[0])
        (d0,v0),(d1,v1)=entries
        if d1<=d0: raise ValueError('Distinct depth labels required')
        segments.append((point,v0,v1,d0,d1))
    base,held=capture_parts(query_path)
    predictions=[infer(spectral_feature(base,s,'shape_delta'),segments) for s in held]
    truth,depth=read_label(query_path)  # scoring only, after inference
    return dict(method='Fixed per-position linear segment between shape features at two measured depths',
        source=str(query_path.resolve()),source_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),truth_point=truth,
        actual_indentation_mm=depth,predictions=predictions,
        correct_frames=sum(r['point']==truth for r in predictions),total_frames=len(predictions),
        median_depth_error_mm=float(np.median([abs(r['estimated_indentation_mm']-depth) for r in predictions])),
        deployed=False,physical_15hz_verified=False,area_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('query',type=Path)
    p.add_argument('--manifest',type=Path,default=Path('../outputs/current_position_depth_training_manifest_20260907.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=evaluate(a.manifest,a.query)
    with a.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))
