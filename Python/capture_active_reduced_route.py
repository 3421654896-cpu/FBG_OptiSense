"""Explicit active-code route; wire indices are route IDs, not 2001-table lookups."""
import argparse
import binascii
import json
import math
from pathlib import Path
import time
import serial
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import sha256_file, now_text
import capture_candidate27_guarded_stream as transport


def prepare(probe_path,cycles,first_delay):
    root=Path(__file__).resolve().parent
    probe=json.loads(probe_path.read_text('utf-8'))
    active,hidden=load_installed_rows()
    selected=[r for i,r in enumerate(active) if i%5 in (1,2,3)]
    if (probe.get('schema')!='active_reduced_route_settled_probe_v1'
            or not probe.get('complete') or not probe.get('disarm_ack') or not probe.get('shutter_ack')
            or probe.get('route_rows')!=selected or probe.get('hidden_predecessor_codes')!=list(hidden)
            or probe.get('guard_before_local')!=1 or probe.get('guard_hold_s')!=.005
            or probe.get('feedback_selectors')!=[0,2] or probe.get('cnc_commands')!=[]
            or probe.get('active_table_sha256')!=sha256_file(root/'mode_tables_from_fullband_2001.json')
            or len(probe.get('samples',[]))!=54):
        raise ValueError('Matching complete active-code qualification required')
    for i,s in enumerate(probe['samples']):
        meter=s['meter']; point=s['point']; row=selected[i%27]
        if (s['repeat']!=i//27+1 or s['local']!=i%27 or not s['passed'] or not point['stable']
                or tuple(point['dac_codes'])!=tuple(row['codes']) or point['ch1_adc_code']>=4000
                or not math.isfinite(meter['wavelength_nm'])
                or abs(meter['wavelength_nm']-row['target_wavelength_nm'])>.05
                or not math.isfinite(meter['power_mw']) or meter['power_mw']<=0
                or not math.isfinite(meter['side_mode_suppression_db'])
                or meter['side_mode_suppression_db']<20):
            raise ValueError('Qualification sample invalid')
    if len(probe.get('guard_events',[]))!=2 or any(
            e['before_local']!=1 or e['repeat']!=i+1
            or e['completed_ns']-e['started_ns']<5_000_000
            for i,e in enumerate(probe['guard_events'])):
        raise ValueError('Guard qualification invalid')
    # Explicit codes are sent on the wire. IDs only enforce ordered route identity.
    rows=[dict(index=2*i+2,codes=r['codes'],source_fullband_index=r['fullband_index'],
        active_mode_order=r['order']) for i,r in enumerate(selected)]
    guard=dict(index=3,codes=list(hidden),before_order=1,hold_us=5000)
    tag=int(time.time_ns()&0x7fffffff) or 1
    packet=transport.command(rows,guard,cycles,tag,guard_before=1,guard_hold_us=5000,
        first_delay_us=first_delay,spacing_us=50)
    return dict(schema='active_reduced_guarded_stream_v1',complete=False,created=now_text(),
        route_rows=rows,guard=guard,cycles_requested=cycles,first_delay_us=first_delay,
        spacing_us=50,route_crc32=binascii.crc32(packet[24:364]),
        expected_firmware=transport.FIRMWARE,command_hex=packet.hex(),frames=[],
        qualification_sha256=sha256_file(probe_path),wire_indices_are_route_ids=True,
        cnc_commands=[],deployed=False,physical_15hz_verified=False,
        disarm_ack=False,shutter_ack=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=256)
    p.add_argument('--first-delay-us',type=int,default=250)
    a=p.parse_args()
    prepared=prepare(a.probe,a.cycles,a.first_delay_us)
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        transport.execute_device(device,a.output,prepared=prepared)
