"""Reproducible CPU-only Makima cache comparison; never opens a device."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import temporary_peak_fit_makima as m
from temporary_test_mode import frame_adc_values


def measure(rows, values):
    durations, results = [], []
    for value in values:
        start = time.perf_counter()
        results.append(m.fit_temporary_frame(rows, value))
        durations.append((time.perf_counter() - start)*1000)
    return results, dict(median_ms=statistics.median(durations),
                         mean_ms=statistics.mean(durations), max_ms=max(durations))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=args.capture.read_bytes()
    payload=json.loads(raw)
    rows=payload['plan']['rows']
    values=[frame_adc_values(f) for f in payload['frames'] if f['sequence']>=16][:64]
    if not values:raise ValueError('No post-warmup frames')
    basis=m._gaussian_shape_basis
    with patch.object(m,'_gaussian_shape_basis',basis.__wrapped__):
        measure(rows,values[:3])
        uncached,old_rate=measure(rows,values)
    measure(rows,values[:3])
    cached,new_rate=measure(rows,values)
    if uncached!=cached:raise AssertionError('Caching changed complete fitting results')
    result=dict(schema='temporary_makima_cpu_cache_comparison_v1',
                hardware_not_accessed=True,board_fps_change_measured=False,
                input=str(args.capture.resolve()),input_sha256=hashlib.sha256(raw).hexdigest(),
                point_count=len(rows),frames=len(values),complete_outputs_identical=True,
                uncached_basis=old_rate,cached_basis=new_rate,
                median_cpu_reduction_fraction=1-new_rate['median_ms']/old_rate['median_ms'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__=='__main__':raise SystemExit(main())
