"""Synthetic transfer evaluation on CRC-verified measured board timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from causal_scan_alignment import align
from audit_f45_contact_transition import validated_gain_identity
from capture_fast_flank_stream import decode_data, decode_end


def verified_times(data):
    if len(data['frames'][0]['records']) == 45:
        validated_gain_identity(data)
    else:
        data_decoder, end_decoder = decode_data, decode_end
        kwargs={}
        if data.get('explicit_spacing_protocol'):
            from f18_spacing_protocol import decode_data as data_decoder, decode_end as end_decoder
            kwargs['spacing_us']=data['requested_spacing_us']
            kwargs['expected_firmware']=data['expected_firmware_version']
        terminal=bytes.fromhex(data['terminal']['wire_hex'])
        tag=int.from_bytes(terminal[4:8],'big')
        end_kwargs={'expected_firmware':kwargs['expected_firmware']} if 'expected_firmware' in kwargs else {}
        end_decoder(terminal,tag=tag,table_crc32=data['table_codes_crc32'],**end_kwargs)
        for saved in data['frames']:
            decoded=data_decoder(bytes.fromhex(saved['wire_hex']),tag=tag,table_crc32=data['table_codes_crc32'],**kwargs)
            for key in ('sequence','cycle_start_us','elapsed_us','records'):
                if decoded[key]!=saved[key]: raise ValueError('Decoded metadata mismatch')
    frames=[f for f in data['frames'] if f['sequence']>=80]
    if len(frames)<32 or any(b['sequence']!=a['sequence']+1 for a,b in zip(frames,frames[1:])):
        raise ValueError('Incomplete sequence')
    return np.array([[f['cycle_start_us']+r['second_end_us'] for r in f['records']] for f in frames],float)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists')
    raw=a.capture.read_bytes(); times=verified_times(json.loads(raw))
    times-=times[0,0]
    gap_limit=1.5*float(np.median(np.diff(times[:,0])))
    rows=[]
    for hz in (1,5,10,15):
        values=np.sin(2*np.pi*hz*times*1e-6)
        results=[align(times[i-1],values[i-1],times[i],values[i],max_gap_us=gap_limit) for i in range(1,len(times))]
        target=np.array([r['target_us'] for r in results])*1e-6
        aligned=np.array([r['values'] for r in results])
        truth=np.sin(2*np.pi*hz*target)[:,None]
        design=np.c_[np.sin(2*np.pi*hz*target),np.cos(2*np.pi*hz*target),np.ones(len(target))]
        coefficients=np.linalg.lstsq(design,aligned,rcond=None)[0]
        amplitudes=np.hypot(coefficients[0],coefficients[1])
        rows.append(dict(frequency_hz=hz,
            unaligned_rms_error_at_frame_start=float(np.sqrt(np.mean((values[1:]-truth)**2))),
            aligned_rms_error=float(np.sqrt(np.mean((aligned-truth)**2))),
            aligned_amplitude_ratio_per_point=amplitudes.tolist(),
            minimum_amplitude_ratio=float(amplitudes.min()),
            median_amplitude_ratio=float(np.median(amplitudes)),
            inherent_delay_median_us=float(np.median([r['inherent_delay_us'] for r in results]))))
    report=dict(capture_sha256=hashlib.sha256(raw).hexdigest(),points=times.shape[1],
        evaluated_scan_pairs=len(times)-1,rows=rows,simulation_only=True,
        live_adc_values_not_used=True,physical_15hz_verified=False,deployed=False,
        scope='Unit-amplitude identical sinusoid on all points at verified completion timestamps. No pressure excitation, no force/area measurement. Alignment uses only the completed current scan and its predecessor, with explicit delay; not a fix qualified for production.')
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='aligned_amplitude_ratio_per_point'} for r in rows]},indent=2))


if __name__=='__main__': main()
