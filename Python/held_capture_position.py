"""Actual CNC position brackets for stationary optical diagnostics, not force truth."""
import math
from displacement_pressure_reference import from_machine


def reference(target, status):
    if not status.get('connected') or status.get('moving') or not status.get('stopped'):
        raise ValueError('Connected stationary machine required')
    if status.get('estop') or status.get('estop_led'):
        raise ValueError('E-stop active')
    if not all(status.get(f'{a}_homed') for a in 'xyz'):
        raise ValueError('Homed XYZ required')
    if any(status.get(f'{a}_{d}_limit') for a in 'xyz' for d in ('pos', 'neg')):
        raise ValueError('Hardware limit active')
    xyz = [float(status[f'machine_{a}']) for a in 'xyz']
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError('Finite actual machine coordinates required')
    pressure = from_machine(target, status)
    if pressure is None or pressure['over_depth_limit']:
        raise ValueError('Missing contact plane or excessive indentation')
    return {'machine_xyz_mm': xyz, 'pressure_reference': pressure}


def bracket(target, before, after, tolerance_mm=0.003):
    a, b = reference(target, before), reference(target, after)
    if any(abs(x-y) > tolerance_mm for x,y in zip(a['machine_xyz_mm'], b['machine_xyz_mm'])):
        raise ValueError('Machine position changed during diagnostic')
    return {'before': a, 'after': b,
            'stationarity_verified_throughout': False,
            'timing_basis': 'status_before_warmup_and_after_teacher_only',
            'pressure_source': 'calibrated_displacement_not_force_sensor'}
