"""Reset only after exact known application and protected digest verification."""
import argparse
import hashlib
import json
from pathlib import Path
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from fbg_ota_protocol import load_package
from guarded_application_upgrade import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',required=True); p.add_argument('--package',type=Path,required=True)
    p.add_argument('--protected-sha256',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--restore-hsi',action='store_true',help='While halted, restore internal clock before readback; no flash writes')
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists')
    pkg=load_package(a.package); report=dict(reset_sent=False,no_flash_writes=True,no_cnc_commands=True)
    try:
        session=ConnectHelper.session_with_chosen_probe(blocking=False,unique_id=a.probe,
            options={'target_override':'cortex_m','connect_mode':'attach','auto_unlock':False,
                     'resume_on_disconnect':False,'no_config':True,'frequency':1000000,
                     'cache.enable_memory':False,'cache.enable_register':False})
        if session is None or session.probe.unique_id!=a.probe: raise RuntimeError('Probe mismatch')
        with session:
            t=session.target; report['state_before']=str(t.get_state())
            if a.restore_hsi:
                if t.get_state()!=Target.State.HALTED: raise RuntimeError('Clock recovery requires halted core')
                if not t.read32(0x40023800)&2: raise RuntimeError('HSI not ready')
                cfgr=t.read32(0x40023808)
                report['clock_cfgr_before']=hex(cfgr)
                report['flash_acr_before']=hex(t.read32(0x40023c00))
                t.write32(0x40023808,cfgr & ~3)
                for _ in range(100):
                    if t.read32(0x40023808)&12==0: break
                else: raise RuntimeError('HSI clock switch did not confirm')
                report['clock_cfgr_after']=hex(t.read32(0x40023808))
            actual=read(t,pkg.link_address,len(pkg.image))
            report['application_matches']=actual==pkg.image
            report['application_sha256']=hashlib.sha256(actual).hexdigest()
            protected=hashlib.sha256(read(t,0x08000000,0x8000)).hexdigest()
            report['protected_sha256']=protected
            if actual!=pkg.image or protected!=a.protected_sha256:
                raise RuntimeError('Known image/protected bytes mismatch; refusing reset')
            t.reset(Target.ResetType.SYSRESETREQ)
            report['reset_sent']=True
            if t.get_state()==Target.State.HALTED:
                pc=int(t.read_core_register('pc'))
                report['reset_pc']=hex(pc)
                if not 0x08000000 <= pc < 0x08040000:
                    raise RuntimeError('Reset PC is outside verified boot/application; refusing resume')
                t.resume()
                report['verified_reset_resumed']=True
            report['state_after']=str(t.get_state())
    finally:
        a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__': main()
