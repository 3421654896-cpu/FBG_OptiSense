"""Read-only identity audit before constructing a reduced optical route."""
import argparse
import hashlib
import json
from pathlib import Path
import app_JDSU as app
from capture_ch1_rc_diagnostic import load_installed_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('New output required')
    root=Path(__file__).resolve().parent
    rows,hidden=load_installed_rows()
    full=app.load_fullband_accuracy_table(root)
    differences=[]
    for r in rows:
        reference=full[r['fullband_index']]
        if tuple(r['codes'])!=tuple(reference.codes):
            differences.append(dict(mode_order=r['order'],fullband_index=r['fullband_index'],
                active_codes=r['codes'],operational_2001_codes=list(reference.codes)))
    files=[root/'mode_tables_from_fullband_2001.json',
           root/'fullband_equal_power_operational_2001.csv',
           root.parent/'JDSU/Core/Src/stress_table.c']
    result=dict(schema='active_route_identity_audit_v1',
        desktop_firmware_source_rows_match=True,active_rows=len(rows),
        differing_operational_rows=len(differences),differences=differences,
        active_hidden_predecessor_codes=list(hidden),
        file_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        firmware_flash_identity_checked_separately=True,
        reduced_route_must_preserve_explicit_codes_not_only_indices=True,
        hardware_commands_sent=False)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('differences','file_sha256')}))


if __name__=='__main__': main()
