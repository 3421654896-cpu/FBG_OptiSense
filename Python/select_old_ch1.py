"""Offline spacing-only selection from a saved dense CH1 spectrum."""
import json
from pathlib import Path
import time
from temporary_test_mode import select_points

def select_old_reference(path,spacing_nm=.16,baseline_fraction=None,soft_baseline=False):
    import app_JDSU as app
    path=Path(path)
    reference=json.loads(path.read_text(encoding='utf-8'))
    rows=reference['rows']
    if not reference.get('complete') or len(rows)!=2001 or [r['index'] for r in rows]!=list(range(2001)):
        raise ValueError('需要完整2001点旧CH1光谱，不使用残缺扫描')
    x=[r['measured_wavelength_nm'] for r in rows]
    y=[r['ch1_adc_code'] for r in rows]
    plan=select_points(app.load_fullband_accuracy_table(),x,y,spacing_nm=spacing_nm,baseline_fraction=baseline_fraction,soft_baseline=soft_baseline)
    plan.update(reference_source=str(path.resolve()),reference_wavelengths_nm=x,reference_values=y,
                selection_only=True,stability_filter_applied=False,spacing_nm=spacing_nm,
                warning='Spacing-only selection; not a new calibration or stability certification')
    return plan

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    config=json.loads((root/'temporary_test_defaults.json').read_text(encoding='utf-8'))
    plan=select_old_reference(root/config['selection_reference'],config['spacing_nm'],config.get('baseline_fraction'),config.get('soft_baseline',False))
    output=root/'outputs/temporary_test'/f'old_ch1_spacing_selection_{time.time_ns()}.json'
    output.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output)
    for group in range(1,10):
        selected=[r for r in plan['rows'] if r['group']==group]
        assert len({b['index']-a['index'] for a,b in zip(selected,selected[1:])})==1
        print(group,[r['index'] for r in selected],[r['target_nm'] for r in selected])

