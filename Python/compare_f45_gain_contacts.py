"""Compare raw ADC response/noise at the same commanded point and depth."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_f45_contact_drift import trace_features, phase_masks
from run_f45_depth_round_visible import verify_capture
from audit_f45_contact_transition import validated_gain_identity
from audit_local_training_geometry import physical_target


def measure(path):
    data, seq, features = trace_features(path)
    c = data['commissioning']
    gain = data['feedback_selectors'][1]
    validated_gain_identity(data,expected_selectors=(0,gain))
    verify_capture(path, c['target']['point_index'], c['depth_mm'], gain)
    masks = phase_masks(data,seq)
    first = features[:,:45]
    base = first[masks['baseline_fit_window']]
    held = first[masks['held']]
    signal = float(np.sqrt(np.mean(np.median(held,axis=0)**2)))
    # Median offset signal vs within-baseline per-row temporal deviation.
    noise = float(np.sqrt(np.mean(np.var(base,axis=0))))
    raw = np.array([[r['first_code'] for r in f['records']] for f in data['frames']])
    grating_response = np.median(held,axis=0).reshape(9,5)
    grating_noise = np.sqrt(np.mean(np.var(base,axis=0).reshape(9,5),axis=1))
    return {'capture':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'point':int(c['target']['point_index']), 'depth_mm':c['depth_mm'],
        'baseline_start_sequence':c.get('baseline_start_sequence',16),
        'noise_basis':'temporal variation in the baseline window; includes startup drift',
        'ch1_kohm':data['ch1_transimpedance_kohm'],
        'firmware_version':data['frames'][0]['firmware_version'],
        'table_crc32':data['table_codes_crc32'],
        'machine_contact_xyz_mm':physical_target(c['target']).tolist(),
        'signal_rms_codes':signal,'noise_rms_codes':noise,
        'signal_to_noise_ratio':signal/max(noise,1e-12),
        'per_grating_signed_response_codes':grating_response.tolist(),
        'per_grating_signal_rms_codes':np.sqrt(np.mean(grating_response**2,axis=1)).tolist(),
        'per_grating_baseline_noise_rms_codes':grating_noise.tolist(),
        'post_release_rms_codes':float(np.sqrt(np.mean(first[masks['after_safe_retraction']]**2))),
        'maximum_raw_code':int(raw.max()),'saturated_samples':int((raw>=4090).sum()),
        'sampling_hz':data['analysis']['board_rate_hz'],
        'sampling_gate':data['analysis']['meets_sampling_rate_for_15hz'],
        'held_frames':int(masks['held'].sum())}


def validate_comparison(rows, depth_series=False):
    if not rows:
        raise ValueError('No captures')
    if len({r['table_crc32'] for r in rows}) != 1:
        raise ValueError('Wavelength tables differ')
    if any(not np.allclose(r['machine_contact_xyz_mm'],rows[0]['machine_contact_xyz_mm'],
                           atol=1e-6,rtol=0) for r in rows):
        raise ValueError('Physical calibration targets differ')
    if depth_series:
        if len({(r['point'],r['ch1_kohm'],r['firmware_version']) for r in rows}) != 1:
            raise ValueError('Depth comparison requires same point, gain and firmware')
    elif len({(r['point'],r['depth_mm']) for r in rows}) != 1:
        raise ValueError('Comparison requires same point and commanded depth')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth-series',action='store_true',
                   help='Compare depths at one point with unchanged gain and firmware')
    a=p.parse_args()
    if a.output.exists(): p.error('output must be new')
    rows=[measure(path) for path in a.capture]
    validate_comparison(rows,a.depth_series)
    result={'schema':'f45_gain_contact_comparison_v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'captures':rows,
        'physical_15hz_verified':False,'position_accuracy_verified':False,
        'force_measured':False,'contact_area_verified':False}
    result['comparison_kind']='depth_series' if a.depth_series else 'gain_series'
    vectors=np.array([r['per_grating_signed_response_codes'] for r in rows]).reshape(len(rows),45)
    norms=np.linalg.norm(vectors,axis=1)
    result['response_shape_cosine']=[[
        float(vectors[i]@vectors[j]/(norms[i]*norms[j]))
        if norms[i]>0 and norms[j]>0 else None
        for j in range(len(rows))] for i in range(len(rows))]
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
