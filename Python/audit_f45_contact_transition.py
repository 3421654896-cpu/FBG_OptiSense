"""Host-timestamp transition audit; commanded-depth events are not force truth."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from audit_f45_contact_drift import trace_features
from audit_f45_contact_hysteresis import hysteresis
from select_f45_activation_threshold_v6 import load_manifest
from capture_fast_flank_stream import _crc_ok


def require_fresh_capture(payload,data,digest):
    if digest in {r['sha256'] for r in payload['captures']}:
        raise ValueError('Validation capture belongs to training')
    if datetime.fromisoformat(data['created_utc'])<=datetime.fromisoformat(payload['created_utc']):
        raise ValueError('Validation capture predates model freeze')


def validated_gain_identity(data, expected_selectors=(0,2)):
    """Recover legacy metadata from CRC-checked v1.0.53/54 USB F45 bytes."""
    expected_selectors = tuple(expected_selectors)
    if len(expected_selectors)!=2 or any(v not in range(4) for v in expected_selectors):
        raise ValueError('invalid expected selectors')
    terminal = bytes.fromhex(data['terminal']['wire_hex'])
    if len(terminal)!=42 or not _crc_ok(terminal) or terminal[:3] != b'\xd9\x9d\x04' or terminal[20:22] != bytes(expected_selectors):
        raise ValueError('terminal gain or integrity mismatch')
    firmware = int.from_bytes(terminal[8:12],'big')
    if firmware not in (0x10035,0x10036):
        raise ValueError('historical firmware protocol not reviewed')
    if 'feedback_selectors' in data and tuple(data['feedback_selectors']) != expected_selectors:
        raise ValueError('metadata contradicts expected gain')
    for saved in data['frames']:
        if len(saved['records'])!=45: raise ValueError('missing decoded records')
        raw = bytes.fromhex(saved['wire_hex'])
        if (len(raw)!=453 or raw[:3]!=b'\xd9\x9d\x03' or not _crc_ok(raw)
                or raw[4:12]!=terminal[4:12] or raw[32:34]!=bytes(expected_selectors)
                or raw[40:85]!=bytes(range(45)) or raw[3]&2
                or int.from_bytes(raw[24:28],'big')!=data['table_codes_crc32']
                or int.from_bytes(raw[12:16],'big')!=saved['sequence']):
            raise ValueError('raw F45 identity mismatch')
        if (int.from_bytes(raw[16:20],'big') != saved['cycle_start_us']
                or int.from_bytes(raw[20:24],'big') != saved['elapsed_us']):
            raise ValueError('decoded cycle timing contradicts wire data')
        previous_end = -1
        for i, sample in enumerate(saved['records']):
            at = 85+i*8
            expected = [int.from_bytes(raw[j:j+2],'big') for j in range(at,at+8,2)]
            actual = [sample[k] for k in ('first_code','second_code','write_end_us','second_end_us')]
            if expected != actual: raise ValueError('decoded samples contradict wire data')
            if not previous_end < sample['write_end_us'] < sample['second_end_us'] <= saved['elapsed_us']+1:
                raise ValueError('non-causal sample timing')
            previous_end = sample['second_end_us']
    return {'selectors':list(expected_selectors),'firmware_version':firmware,'basis':'all raw F45 frames and terminal CRC checked'}


def transition(times, prediction, events, point):
    times = np.asarray(times,dtype=np.int64)
    pred = np.asarray(prediction)
    if len(times) != len(pred) or np.any(np.diff(times)<0):
        raise ValueError('invalid frame receipt series')
    depth = events['contact_reached_commanded_depth']
    release = events['release_started']
    safe = events['safe_height_reached']
    if not events['contact_descent_started'] < depth < release < safe:
        raise ValueError('invalid event order')
    masks = {'first_200ms_after_depth':(times>=depth)&(times<min(depth+200_000_000,release)),
             'whole_depth_hold':(times>=depth)&(times<release),
             'after_safe_retraction':times>=safe}
    phases = {}
    for name, mask in masks.items():
        expected = 0 if name == 'after_safe_retraction' else point
        rows = np.flatnonzero(mask)
        correct = rows[pred[rows]==expected]
        origin = safe if expected==0 else depth
        phases[name] = {'frames':len(rows),'correct':len(correct),
                       'first_correct_receipt_after_event_ms':float((times[correct[0]]-origin)/1e6) if len(correct) else None}
    return phases


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--round',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-fresh',action='store_true')
    a = p.parse_args()
    payload = joblib.load(a.model)
    model = payload['model']
    contact = model.classes_ != 0
    results, excluded = [], []
    for manifest in a.round:
        for item in load_manifest(manifest):
            data, seq, x = trace_features(item['path'])
            if a.require_fresh:
                require_fresh_capture(payload,data,hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
            try:
                identity = validated_gain_identity(data)
                if not data['complete']: raise ValueError('incomplete capture')
            except (ValueError, KeyError) as exc:
                excluded.append({'source':str(item['path']),
                                 'reason':str(exc)})
                continue
            c = data['commissioning']
            receipts = {r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
            events = {e['kind']:e['host_monotonic_ns'] for e in c['events']}
            valid = seq >= c.get('baseline_start_sequence',16)+64
            prob = model.predict_proba(x[valid])[:,contact]
            state = hysteresis(prob.sum(axis=1),.4,.3)
            pred = np.where(state,model.classes_[contact][prob.argmax(axis=1)],0)
            times = [receipts[int(s)] for s in seq[valid]]
            t = np.asarray(times,dtype=np.int64)
            diagnostic = {}
            for name, mask in {
                'pre_descent':t<events['contact_descent_started'],
                'depth_hold':(t>=events['contact_reached_commanded_depth'])&(t<events['release_started']),
                'after_retraction':t>=events['safe_height_reached']}.items():
                if not mask.any(): continue
                labels, counts = np.unique(pred[mask],return_counts=True)
                votes = model.classes_[contact][prob[mask].argmax(axis=1)]
                vote_labels,vote_counts = np.unique(votes,return_counts=True)
                confidence = prob[mask].sum(axis=1)
                rms = np.sqrt(np.mean(x[valid][mask,45:90]**2,axis=1))
                diagnostic[name] = {'predictions':dict(zip(map(str,labels),map(int,counts))),
                    'location_votes_before_contact_gate':dict(zip(map(str,vote_labels),map(int,vote_counts))),
                    'contact_probability_min_median_max':[float(confidence.min()),float(np.median(confidence)),float(confidence.max())],
                    'second_adc_change_rms_min_median_max':[float(rms.min()),float(np.median(rms)),float(rms.max())]}
            results.append({'point':item['point'],'round':item['round'],
                            'gain_identity':identity,
                            'failure_diagnostic':diagnostic,
                            'source':str(item['path']),
                            'phases':transition(times,pred,events,item['point'])})
    report = {'schema':'f45_transition_receipt_audit_v1','results':results,'excluded':excluded,
              'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
              'physical_15hz_verified':False,'new_blind_test':False,
              'prospective_capture_and_training_disjointness_checked':a.require_fresh,
              'limitation':'Host receipt times, not ADC exposure or initial contact times; mechanical endpoint uses Mach3 report, not force sensor. First 200ms retained, no settling exclusion.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(results))
