"""Audit same-trial production ADC/RC against a held, direct-ADC teacher.

Read-only analysis: no model fitting, device control or deployment. Optical
agreement does not certify physical pressure bandwidth, location or area.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from app_JDSU import fullband_accuracy_statistics
from capture_production_pair_teacher import GAIN_KOHM, validate_capture_profile, selected_points, decode as decode_pair_packet

METHODS = ('first_direct', 'second_direct', 'last_production_direct',
           'firmware_rc', 'post_pair_direct')


def rc_feasibility(first, second, teacher):
    """Can a monotonic, single-pole exponential reach this same-trial teacher?

    alpha=(second-teacher)/(first-teacher). This is a diagnostic derived from
    a future label, never a causal prediction or fitted runtime coefficient.
    Even a compatible pair does not prove the laser input stayed constant.
    """
    if first == second == teacher:
        return {'classification': 'already_at_teacher', 'required_alpha': None}
    if first == teacher:
        return {'classification': 'incompatible_with_single_pole', 'required_alpha': None}
    alpha = float((second-teacher)/(first-teacher))
    return {'classification': ('compatible_with_single_pole' if 0 <= alpha < 1
                               else 'incompatible_with_single_pole'),
            'required_alpha': alpha}


def finite_code(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 4095:
        raise ValueError('invalid ADC code')
    return float(value)


def checked_record(r):
    if 'wire_packet_hex' in r:
        decoded=decode_pair_packet(bytes.fromhex(r['wire_packet_hex']),r['point_index'],
                                   r['request_tag'],r['feedback_selectors'][1],r['schedule_version'])
        if any(r.get(key)!=value for key,value in decoded.items()):
            raise ValueError('stored capture differs from original wire packet')
    if (r.get('same_trial') is not True or
            r.get('diagnostic_interrupts_scan') is not True or
            r.get('firmware_shutter_confirmed') is not True or
            r.get('pressure_ground_truth') is not None):
        raise ValueError('record provenance does not match optical-only capture')
    t = r['teacher']
    if (t.get('dac_rewritten') is not False or t.get('adc_value_kind') != 'direct_adc' or
            t['channels'] != ['PDT', 'PDR', 'CH0', 'CH1', 'CH2', 'CH3']):
        raise ValueError('teacher is not same-held direct ADC')
    version, point = r['schedule_version'], r['point_index']
    selectors = r['feedback_selectors']
    if (version not in (4, 5) or r['firmware_version'] < 0x1001e or
            not 0 <= point < 45 or point // 5 + 1 != r['grating'] or
            len(selectors) != 2 or selectors[0] != selectors[1] or
            selectors[1] not in GAIN_KOHM or point not in r['planned_indices']):
        raise ValueError('invalid capture condition')
    reads = r['production_reads']
    if [x['kind'] for x in reads] not in (
            ['first', 'second'], ['first', 'second', 'slow_recheck']):
        raise ValueError('invalid production read order')
    end = -1
    for x in reads:
        if not end < x['start_us'] < x['end_us'] < 300000:
            raise ValueError('invalid production timestamps')
        finite_code(x['direct_ch1'])
        end = x['end_us']
    if r['pair_available_us'] != end:
        raise ValueError('availability differs from last production read')
    optical = r['optical_state']
    teachers = r['teacher_samples']
    if len(teachers) != 25:
        raise ValueError('teacher must have exactly 25 scheduled groups')
    for i, x in enumerate([optical] + teachers):
        if not end < x['start_us'] < x['end_us'] <= 850000:
            raise ValueError('teacher/feature time order invalid')
        if i == 0:
            if x['end_us'] >= 300000 or r['extended_features_available_us'] != x['end_us']:
                raise ValueError('future teacher used as early optical feature')
        elif not 300000 + (i-1)*20000 <= x['start_us'] <= 310000 + (i-1)*20000:
            raise ValueError('teacher not in scheduled window')
        if len(x['direct_adc']) != 6:
            raise ValueError('invalid six-channel ADC group')
        for v in x['direct_adc']:
            finite_code(v)
        end = x['end_us']
    ordered = [[x['direct_adc'][c] for c in (4, 5, 0, 1, 2, 3)] for x in teachers]
    median, sigma, drift, stable = fullband_accuracy_statistics(ordered)
    for key, expected in [('median', median), ('sigma', sigma), ('drift', drift)]:
        if not np.allclose(t[key], expected, rtol=0, atol=1e-9):
            raise ValueError('stored teacher statistics disagree with raw samples')
    if t['stable'] is not bool(stable):
        raise ValueError('stored teacher qualification disagrees with raw samples')
    reference = float(median[3])
    values = dict(zip(METHODS, [reads[0]['direct_ch1'], reads[1]['direct_ch1'],
                              reads[-1]['direct_ch1'], r['firmware_rc_estimate'],
                              optical['direct_adc'][1]]))
    for v in values.values():
        finite_code(v)
    return {
        'schedule_version': version, 'grating': r['grating'], 'point_index': point,
        'selector': selectors[1], 'gain_kohm': GAIN_KOHM[selectors[1]],
        'repeat': r['repeat'], 'request_tag': r['request_tag'],
        'profile': r['profile'], 'primary_segment': r['primary_segment'],
        'secondary_segment': r['secondary_segment'],
        'teacher_code': reference, 'teacher_stable': bool(stable),
        'teacher_ch1_sigma': float(sigma[3]), 'teacher_ch1_drift': float(drift[3]),
        'read_count': len(reads), 'pair_available_us': r['pair_available_us'],
        'extended_features_available_us': r['extended_features_available_us'],
        'values': values, 'signed_errors': {m: float(v-reference) for m, v in values.items()},
        'diagnostic_tolerance_codes': max(3.0, reference * .10),
        'single_pole_diagnostic': rc_feasibility(reads[-2]['direct_ch1'], reads[-1]['direct_ch1'], reference),
    }


def summarize(rows):
    qualified = [r for r in rows if r['teacher_stable']]
    result = {'total': len(rows), 'qualified_teacher_count': len(qualified),
              'excluded_unstable_teacher_count': len(rows)-len(qualified), 'methods': {}}
    if not qualified:
        return result
    tolerance = np.array([r['diagnostic_tolerance_codes'] for r in qualified])
    refs = np.array([r['teacher_code'] for r in qualified])
    for method in METHODS:
        err = np.array([r['signed_errors'][method] for r in qualified])
        absolute = np.abs(err)
        result['methods'][method] = {
            'count': len(err), 'bias_codes': float(np.mean(err)),
            'mae_codes': float(np.mean(absolute)), 'rmse_codes': float(np.sqrt(np.mean(err**2))),
            'max_abs_error_codes': float(np.max(absolute)),
            'within_3_codes_count': int(np.sum(absolute <= 3)),
            'within_diagnostic_tolerance_count': int(np.sum(absolute <= tolerance)),
            'relative_mae_nonzero_teacher': float(np.mean(absolute[refs>0]/refs[refs>0])) if np.any(refs>0) else None,
        }
    result['pair_availability_us'] = {'min': min(r['pair_available_us'] for r in qualified),
                                       'max': max(r['pair_available_us'] for r in qualified)}
    result['teacher_range_codes'] = [float(min(refs)), float(max(refs))]
    result['single_pole_diagnostic_counts'] = {
        name: sum(r['single_pole_diagnostic']['classification'] == name for r in qualified)
        for name in ('already_at_teacher', 'compatible_with_single_pole', 'incompatible_with_single_pole')}
    return result


def shape_diagnostics(wavelengths, teacher, predicted):
    """Five-point optical shape diagnostics, NOT an exact Bragg peak fit."""
    x,t,p=(np.asarray(a,dtype=float) for a in (wavelengths,teacher,predicted))
    if any(a.shape!=(5,) or not np.all(np.isfinite(a)) for a in (x,t,p)) or np.any(np.diff(x)<=0):
        raise ValueError('shape requires five finite, increasing wavelength rows')
    span=float(np.ptp(t)); pspan=float(np.ptp(p))
    result={'teacher_span_codes':span,'prediction_span_codes':pspan,
            'rmse_codes':float(np.sqrt(np.mean((p-t)**2))),
            'shape_resolved':span>=3 and pspan>=3,
            'normalized_shape_rmse':None,'coarse_centroid_shift_pm':None,
            'note':'Independently held points; baseline-subtracted five-point shape and coarse centroid, not simultaneous peak centre or pressure accuracy.'}
    if result['shape_resolved']:
        tn=(t-t.min())/span; pn=(p-p.min())/pspan
        # Work in pm around the middle row to avoid subtracting two ~1550 nm
        # centroid values. Integration acknowledges nonuniform wavelength gaps.
        xp=(x-x[2])*1000
        centroid=lambda y:float(np.trapezoid(xp*y,xp)/np.trapezoid(y,xp))
        result['normalized_shape_rmse']=float(np.sqrt(np.mean((pn-tn)**2)))
        result['coarse_centroid_shift_pm']=centroid(pn)-centroid(tn)
    return result


def reference_shapes(doc, rows):
    if doc.get('full_map_reference') is not True:
        return []
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['schedule_version'],row['selector'],row['repeat'],row['grating'])].append(row)
    results=[]
    for (version,selector,repeat,grating), group in sorted(grouped.items()):
        group=sorted(group,key=lambda r:r['point_index'])
        indices=[r['point_index'] for r in group]
        complete=indices==list(range((grating-1)*5,grating*5))
        entry={'schedule_version':version,'selector':selector,'repeat':repeat,'grating':grating,
               'point_indices':indices,'complete_five_points':complete,
               'all_teachers_qualified':complete and all(r['teacher_stable'] for r in group)}
        entry['wavelengths_nm']=[float(doc['rows'][p]['measured_wavelength_nm']) for p in indices]
        entry['teacher_codes']=[r['teacher_code'] for r in group]
        entry['firmware_rc_codes']=[r['values']['firmware_rc'] for r in group]
        entry['second_direct_codes']=[r['values']['second_direct'] for r in group]
        entry['shape_metrics']={}
        if entry['all_teachers_qualified']:
            for method in ('firmware_rc','second_direct'):
                entry['shape_metrics'][method]=shape_diagnostics(
                    entry['wavelengths_nm'],entry['teacher_codes'],[r['values'][method] for r in group])
        results.append(entry)
    return results


def analyze(doc):
    safety = doc.get('safety_cleanup', {})
    if (doc.get('schema') != 'production_pair_same_hold_teacher_v1' or
            doc.get('trained_model') is not False or
            safety.get('exact_disarm_ack') is not True or
            safety.get('exact_soa_shutter_ack') is not True):
        raise ValueError('capture provenance or safe cleanup missing')
    rows = [checked_record(r) for r in doc['captures']]
    if not rows:
        raise ValueError('no captured records')
    keys = [(r['schedule_version'], r['selector'], r['point_index'], r['repeat']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate experimental condition/repetition')
    if doc.get('full_map_reference') is True:
        for record in doc['captures']:
            validate_capture_profile(record,full_map_reference=True)
        expected={(v,s,p,r) for v in doc['schedule_versions']
                  for s in doc['requested_selectors'] for g in doc['requested_gratings']
                  for p in selected_points(g,v,full_map_reference=True)
                  for r in range(doc['requested_repeats'])}
        if not set(keys)<=expected or (doc.get('complete') is True and set(keys)!=expected):
            raise ValueError('full-reference completeness contradicts requested point coverage')
    groups, aggregate = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[f"v{row['schedule_version']}_{row['gain_kohm']}k_G{row['grating']}_p{row['point_index']}"].append(row)
        aggregate[f"v{row['schedule_version']}_{row['gain_kohm']}k"].append(row)
    return {
        'schema': 'production_pair_teacher_audit_v1',
        'source_capture_complete': doc.get('complete') is True,
        'full_map_reference': doc.get('full_map_reference',False),
        'scope': __doc__, 'trained_model': False, 'deployment_qualified': False,
        'limitations': [
            'Unstable teachers excluded from error summaries, retained in records.',
            'Tolerance max(3 ADC codes, 10% teacher) is descriptive, not user acceptance.',
            'Repeated no-commanded-motion trials do not span independently labelled pressure.',
            'Holding 0.3..0.78 s interrupts scanning; not a physical 15 Hz test.',
            'Post-pair six-channel feature is not present in normal production frames.',
            'Neither ADC error nor FBG response spread is ground-truth position/contact area.',
            'No per-point constant or future teacher is fitted into a predictor.',
            'Single-pole feasibility uses exact integer samples without noise tolerance; descriptive only.',
        ],
        'aggregate': {k: summarize(v) for k,v in sorted(aggregate.items())},
        'groups': {k: summarize(v) for k,v in sorted(groups.items())}, 'records': rows,
        'reference_shapes':reference_shapes(doc,rows),
    }


def plot_reference_shapes(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shapes=report['reference_shapes']
    if not shapes:raise ValueError('No explicit full-reference shapes available')
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for grating,ax in enumerate(axes.flat,1):
        for row in (r for r in shapes if r['grating']==grating):
            if not row['all_teachers_qualified']:continue
            x=np.asarray(row['wavelengths_nm']); x=(x-x[2])*1000
            for key,color,label in [('teacher_codes','#dc8d21','Held direct ADC'),
                                    ('firmware_rc_codes','#2773c1','Fast RC estimate')]:
                ax.plot(x,row[key],'-o',color=color,alpha=.65,markersize=3,
                        label=label if row['repeat']==0 else None)
        ax.set_title(f'G{grating}');ax.grid(alpha=.15)
        ax.set_xlabel('Table wavelength offset (pm)');ax.set_ylabel('CH1 ADC code')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Independent same-DAC holds: early RC vs held direct ADC\nNot a simultaneous spectrum or physical pressure validation',fontsize=13)
    fig.savefig(output,dpi=140);plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot',type=Path)
    args = p.parse_args()
    data = args.input.read_bytes()
    report = analyze(json.loads(data))
    report['source'] = str(args.input.resolve())
    report['source_sha256'] = hashlib.sha256(data).hexdigest()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.plot:plot_reference_shapes(report,args.plot)
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
