"""Separate rate, within-run noise, repeat drift and settled-reference error."""
import argparse
import json
from pathlib import Path
import numpy as np
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file


def load(path):
    report=json.loads(path.read_text('utf-8'))
    if not report.get('complete') or not report.get('disarm_ack') or not report.get('shutter_ack'):
        raise ValueError('Incomplete or unconfirmed optical cleanup: '+str(path))
    return report


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float)**2)))


def analyze(before_path, after_path, paths):
    before,after=load(before_path),load(after_path)
    rows=before['route_rows']
    if len(rows)!=27 or rows!=after['route_rows'] or before['feedback_selectors']!=[0,2] or after['feedback_selectors']!=[0,2]:
        raise ValueError('Reference rows/feedback do not match')
    reference=[]
    for source in (before,after):
        samples=source['samples']
        if len(samples)!=54 or any(s['order']!=i%27 or s['point']['index']!=rows[i%27]['index'] for i,s in enumerate(samples)):
            raise ValueError('Settled reference sequence mismatch')
        reference.append(np.array([s['point']['ch1_adc_code'] for s in samples]).reshape(2,27))
    centers=[np.median(r,axis=0) for r in reference]
    report=dict(schema='candidate27_timing_audit_v1', deployed=False, physical_15hz_verified=False,
                location_accuracy_verified=False, before_sha256=sha256_file(before_path),
                after_sha256=sha256_file(after_path),
                caveat='Sequential captures, not simultaneous; reference drift is not sampling-timing error.',
                settled_before_to_after_rms_codes=rms(centers[1]-centers[0]),
                settled_within_two_round_rms_codes=[rms(r[1]-r[0]) for r in reference],
                captures=[], repeats=[])
    medians=[]
    for path in paths:
        source=load(path)
        if rows!=source['route_rows'] or len(source['frames'])!=source['cycles_requested']:
            raise ValueError('Capture route/length mismatch')
        frames=source['frames'][16:]
        if len(frames)<16 or any(f['sequence']!=i+16 or f['i2c_error_delta'] or f['spi_error_delta'] for i,f in enumerate(frames)):
            raise ValueError('Capture errors or missing fresh cycles')
        item=dict(path=str(path.resolve()),sha256=sha256_file(path),
                  first_delay_us=source['first_delay_us'],spacing_us=source['spacing_us'],
                  rate_hz=source['analysis']['board_rate_hz'],
                  maximum_cycle_us=source['analysis']['board_interval_max_us'],
                  samples=[])
        for field in ('first_code','second_code'):
            data=np.array([[r[field] for r in f['records']] for f in frames],dtype=float)
            median=np.median(data,axis=0)
            item['samples'].append(dict(field=field,medians=median.tolist(),
                rms_to_before_teacher_codes=rms(median-centers[0]),
                rms_to_after_teacher_codes=rms(median-centers[1]),
                max_error_after_teacher_codes=float(np.max(np.abs(median-centers[1]))),
                group_rms_to_after_teacher_codes=[rms(g) for g in (median-centers[1]).reshape(9,3)],
                within_run_rms_noise_codes=rms(np.std(data,axis=0)),
                first_to_last_quarter_rms_drift_codes=rms(np.median(data[-len(data)//4:],axis=0)-np.median(data[:len(data)//4],axis=0))))
        medians.append(item)
        report['captures'].append(item)
    for i,a in enumerate(medians):
        for b in medians[i+1:]:
            if (a['first_delay_us'],a['spacing_us'])!=(b['first_delay_us'],b['spacing_us']): continue
            report['repeats'].append(dict(first=a['path'],second=b['path'],
                first_read_rms_codes=rms(np.array(a['samples'][0]['medians'])-b['samples'][0]['medians']),
                second_read_rms_codes=rms(np.array(a['samples'][1]['medians'])-b['samples'][1]['medians'])))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',required=True,type=Path)
    parser.add_argument('--after',required=True,type=Path)
    parser.add_argument('--capture',required=True,type=Path,nargs='+')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report=analyze(args.before,args.after,args.capture)
    save_checkpoint(args.output,report)
    print('settled-reference shift RMS codes:',report['settled_before_to_after_rms_codes'])
    for item in report['captures']:
        print(Path(item['path']).name,'Hz',round(item['rate_hz'],3),
              'first/second RMS to after teacher',
              [round(s['rms_to_after_teacher_codes'],3) for s in item['samples']])
    print('repeats:',report['repeats'])
