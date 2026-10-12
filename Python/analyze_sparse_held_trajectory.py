"""Offline causal/sampled-settling audit of sparse-prefix diagnostic captures.

No model fit or deployment. The suffix bound only covers sampled instants up
to 100 ms, not continuous settling or physical pressure bandwidth.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_record(record, horizons=(300,750,2000,3000)):
    teacher = record['teacher']
    if (not teacher['stable'] or teacher['adc_value_kind'] != 'direct_adc'
            or teacher['dac_rewritten'] or not record['firmware_shutter_confirmed']):
        raise ValueError('requires qualified direct held teacher and safe completion')
    if teacher['channels'] != ['PDT','PDR','CH0','CH1','CH2','CH3']:
        raise ValueError('teacher channel mapping mismatch')
    value = float(teacher['median'][3])
    if not np.isfinite(value): raise ValueError('nonfinite teacher')
    records = record['curve']
    if len(records) != 33: raise ValueError('incomplete transient curve')
    previous_end = -1
    for sample in records:
        if not previous_end < sample['start_us'] < sample['end_us']:
            raise ValueError('nonmonotonic conversion interval')
        if not 0 <= sample['direct_adc'][1] <= 4095:
            raise ValueError('invalid direct CH1 value')
        previous_end = sample['end_us']
    errors = []
    for horizon in horizons:
        eligible = [r for r in records if r['end_us'] <= horizon]
        sample = eligible[-1] if eligible else None
        errors.append({'horizon_us':horizon,
                       'sample_end_us':sample['end_us'] if sample else None,
                       'direct_ch1':sample['direct_adc'][1] if sample else None,
                       'error_codes':sample['direct_adc'][1]-value if sample else None})
    tolerance = max(3.0,.1*abs(value))
    bad = [i for i,r in enumerate(records) if abs(r['direct_adc'][1]-value) > tolerance]
    candidate = bad[-1]+1 if bad else 0
    suffix_bound = records[candidate]['end_us'] if candidate < len(records) else None
    values = np.asarray([r['direct_adc'][1] for r in records], dtype=float)
    low, high = min(values[0], value), max(values[0], value)
    excursion = np.maximum(np.maximum(low-values, values-high), 0)
    worst = int(np.argmax(excursion))
    # A fixed-equilibrium one-pole response stays between its initial value
    # and equilibrium. This is an observed shape diagnostic, not a fitted
    # model score or a proof of the physical source of the transient.
    shape = {'initial_sample_ch1':float(values[0]),
             'sampled_min_ch1':float(values.min()),
             'sampled_max_ch1':float(values.max()),
             'initial_to_teacher_interval_excursion_codes':float(excursion[worst]),
             'worst_excursion_end_us':records[worst]['end_us'],
             'exceeds_descriptive_tolerance':bool(excursion[worst] > tolerance)}
    return {'point_index':record['point_index'],'prefix_profile':record['prefix_profile'],
            'schedule_version':record.get('schedule_version',4),
            'teacher_ch1':value,'early_direct_errors':errors,'transient_shape':shape,
            'descriptive_tolerance_codes':tolerance,
            'sampled_suffix_within_tolerance_from_us':suffix_bound,
            'target_normally_sampled':record['target_would_normally_be_sampled']}


def analyze(paths):
    sources, results, excluded = [], [], []
    for path in paths:
        content = path.read_bytes()
        document = json.loads(content)
        if document['schema'] != 'sparse_held_trajectory_v1' or not document['complete']:
            raise ValueError('incomplete or unsupported capture file')
        safe = document['safety_cleanup']
        if not safe['exact_disarm_ack'] or not safe['exact_soa_shutter_ack']:
            raise ValueError('capture file lacks verified safe completion')
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(content).hexdigest()})
        for record in document['captures']:
            try: result = audit_record(record)
            except ValueError as exc:
                excluded.append({'source':str(path),'tag':record['request_tag'],'reason':str(exc)})
                continue
            result['source'] = str(path)
            result['request_tag'] = record['request_tag']
            results.append(result)
    groups = defaultdict(list)
    for result in results: groups[(result['schedule_version'],result['point_index'])].append(result)
    summaries = []
    for (version,point), records in sorted(groups.items()):
        errors = [r['early_direct_errors'][1]['error_codes'] for r in records]
        errors = [e for e in errors if e is not None]
        bounds = [r['sampled_suffix_within_tolerance_from_us'] for r in records]
        summaries.append({'point_index':point,'grating':point//5+1,'captures':len(records),
                          'schedule_version':version,
                          'teacher_range_codes':[min(r['teacher_ch1'] for r in records),max(r['teacher_ch1'] for r in records)],
                          'direct_750us_error_range_codes':[min(errors),max(errors)] if errors else None,
                          'sampled_suffix_bounds_us':bounds,
                          'all_sampled_suffixes_found':all(b is not None for b in bounds)})
    return {'schema':'sparse_held_causal_audit_v1','sources':sources,'records':results,
            'by_point':summaries,'excluded':excluded,
            'scope':'static direct ADC vs same-held teacher; no trained predictor; NOT 15 Hz pressure/position/area accuracy',
            'tolerance_policy':'descriptive only: max(3 ADC codes, 10% of held CH1); NOT acceptance criterion',
            'timing_note':'use conversion-group END for causal availability; no future interpolation. Suffix covers only sampled curve instants <=~100 ms.',
            'legacy_field_note':'v1.0.28 target_write_end_frame_offset_us in early files is header-build time, not exact DAC-write time; unused here.'}


def plot_g8(document, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft YaHei','DejaVu Sans']
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors = {36:'#dd5268',37:'#bd952c',38:'#1b9e9b'}
    for point in (36,37,38):
        records = [r for r in document['captures'] if r['point_index']==point]
        for i,record in enumerate(records):
            curve = record['curve']
            axes[0].plot([r['end_us']/1000 for r in curve],
                         [r['direct_adc'][1] for r in curve],color=colors[point],
                         alpha=.6,label=f'点 {point} / G8 '+{36:'左侧',37:'中心',38:'右侧'}[point] if i==0 else None)
    row = next(r for r in document['captures'] if r['point_index']==36)
    teacher_map = {1:3,4:0,5:1}
    for channel,name,color in [(1,'CH1','#dd5268'),(4,'PDT','#4169c2'),(5,'PDR','#1b9e9b')]:
        reference = row['teacher']['median'][teacher_map[channel]]
        axes[1].plot([r['end_us']/1000 for r in row['curve']],
                     [r['direct_adc'][channel]/reference for r in row['curve']],label=name,color=color)
    axes[1].axhline(1,color='#b3b8c3',linewidth=1,label='稳定教师值')
    for ax in axes:
        ax.set_xlim(0,5); ax.set_xlabel('进入目标点后，ADC 组完成时间 / ms')
        ax.grid(alpha=.16); ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('CH1 直接 ADC / 码')
    axes[1].set_ylabel('读数 / 同目标稳定教师值')
    axes[0].set_title('实际快速扫描路径下：G8 三个候选点')
    axes[1].set_title('G8 左侧点：CH1 与内部 PD 同时变化')
    fig.suptitle('静态诊断：不是压力动态带宽或定位准确率',fontsize=11)
    fig.savefig(output,dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot-g8',type=Path)
    args = p.parse_args()
    report = analyze(args.inputs)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in report['by_point']: print(json.dumps(row),flush=True)
    if args.plot_g8:
        document = json.loads(args.inputs[0].read_bytes())
        plot_g8(document,args.plot_g8)


if __name__ == '__main__': main()
