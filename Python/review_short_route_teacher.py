"""Offline plot of actual short-route pairs versus same-trial later holds.

Re-decodes all raw packets. Does not fit, train, operate devices or certify
wavelength, physical pressure bandwidth, location or true contact area.
"""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from capture_short_route_teacher import summarize
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file


def review(source, output):
    source, output = Path(source), Path(output)
    png = output.with_suffix('.png')
    if output.exists() or png.exists():
        raise FileExistsError(output)
    report = json.loads(source.read_text('utf-8'))
    result = summarize(report)
    trials = result['trials']
    result['source_sha256'] = sha256_file(source)
    result['reviewer_sha256'] = sha256_file(Path(__file__))
    result['summary'] = dict(
        trials=len(trials), direct_pair_records=sum(len(c['decoded']['points']) for c in report['captures']),
        later_six_adc_records=sum(len(c['decoded']['teacher_samples']) for c in report['captures']),
        later_short_windows_stable=sum(t['short_teacher_stable'] for t in trials),
        mean_absolute_difference_codes=statistics.fmean(abs(t['same_trial_error_codes']) for t in trials),
        maximum_absolute_difference_codes=max(abs(t['same_trial_error_codes']) for t in trials),
        maximum_same_point_gap_us=max(t['same_point_gap_max_us'] for t in trials),
        note='Differences include unstable later holds; neither short stability nor repeat agreement certifies physical truth.')
    if set(report['targets']) != {p for n in range(9) for p in (n*5+1,n*5+3)} or report['repeats'] != 2:
        raise ValueError('plot requires all nine two-flank pairs and both actual repetitions')
    figure, axes = plt.subplots(3,3,figsize=(13.5,9),layout='constrained')
    for grating, axis in enumerate(axes.flat):
        selected = [next(t for t in trials if t['target']==target and t['repeat']==repeat)
                    for repeat in range(2) for target in (grating*5+1,grating*5+3)]
        x = range(4)
        axis.bar([i-.18 for i in x],[t['same_trial_second'] for t in selected],.36,
                 color='#2c72b9',label='Fast second ADC')
        held = axis.bar([i+.18 for i in x],[t['same_trial_teacher'] for t in selected],.36,
                        color='#f0a344',label='Same-trial later hold')
        for bar, trial in zip(held, selected):
            if not trial['short_teacher_stable']:
                bar.set_hatch('///'); bar.set_edgecolor('#7f4314')
        axis.set_title(f'G{grating+1} | maximum difference {max(abs(t["same_trial_error_codes"]) for t in selected):.0f} codes')
        axis.set_xticks(list(x),['Left #1','Right #1','Left #2','Right #2'])
        axis.set_ylabel('CH1 ADC code / 5 kOhm')
        axis.grid(axis='y',alpha=.18); axis.set_axisbelow(True)
    handles, labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='outside lower center',ncol=2)
    figure.suptitle('Actual continuous 18-point route: faster sampling is not yet stable-value accuracy\n'
        f'8 cycles per trial; G8 right +1.5 ms; worst same-point gap {result["summary"]["maximum_same_point_gap_us"]/1000:.3f} ms\n'
        'Hatched holds failed short-window stability. No synchronized wavelength or physical pressure/area truth.',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(png,dpi=140)
    plt.close(figure)
    result['plot_sha256'] = sha256_file(png)
    save_checkpoint(output,result)
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    review(args.source,args.output)
