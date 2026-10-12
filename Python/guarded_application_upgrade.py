"""Exact-image CMSIS application upgrade; never mass erase or touch CNC."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import serial
from serial.tools import list_ports
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from pyocd.flash.loader import MemoryLoader
from fbg_ota_protocol import load_package
from benchmark_stress_realtime import _safe_disarm, _close_shutter
import app_JDSU as app

START=0x08008000
END=0x08040000


def validate_sector(base, size):
    if size <= 0 or base < START or base+size > END:
        raise ValueError('Erase sector crosses protected boundary')


def digest(data): return hashlib.sha256(data).hexdigest()


def prepare_flash_clock(target):
    if target.get_state()!=Target.State.HALTED:
        raise RuntimeError('Flash clock preparation requires halted CPU')
    if not target.read32(0x40023800)&2:
        raise RuntimeError('HSI is not ready; refusing flash algorithm')
    before=target.read32(0x40023808)
    target.write32(0x40023808,before & ~3)
    for _ in range(100):
        after=target.read32(0x40023808)
        if after & 12 == 0:
            return dict(cfgr_before=hex(before),cfgr_after=hex(after),hsi_confirmed=True)
    raise RuntimeError('Internal-clock switch unconfirmed; refusing flash algorithm')


def read(target, address, length):
    result=bytearray()
    for offset in range(0,length,1024):
        result.extend(target.read_memory_block8(address+offset,min(1024,length-offset)))
    return bytes(result)


def zero_all_dacs_and_verify(device, timeout_s=3.0):
    """Write real zero source codes before gating SOA for an upgrade."""
    expected=(0,0,0,0,0)
    command=app.build_single_value_dac_command(
        expected,soa_mode=app.PI11210_SOA_SOURCE_MODE)
    device.reset_input_buffer()
    device.write(command)
    device.flush()
    deadline=time.monotonic()+timeout_s
    buffer=bytearray()
    while time.monotonic()<deadline:
        data=device.read(min(max(int(device.in_waiting or 0),20),4096))
        if not data: continue
        buffer.extend(data)
        for frame in app.extract_single_value_frames(buffer):
            if frame[:4]!=b'\xff\xff\x02\x00': continue
            returned=tuple((frame[4+2*i]<<8)|frame[5+2*i] for i in range(5))
            flags=int(frame[15])
            if (returned==expected
                    and frame[14]==app.PI11210_SOA_SOURCE_MODE
                    and (flags&0x87)==0x87 and not (flags&0x30)):
                return True
            raise RuntimeError('All-zero source command was not acknowledged')
    raise RuntimeError('All-zero source command timed out')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',required=True)
    p.add_argument('--port',default='COM6')
    p.add_argument('--device-serial',required=True,
                   help='Exact STM32 USB CDC serial number expected on --port')
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--recover-failed-report',type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists')
    old,new=load_package(a.current),load_package(a.candidate)
    if old.link_address!=START or new.link_address!=START or not 0<len(new.image)<=END-START:
        raise ValueError('Unexpected image bounds')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),apply_requested=a.apply,
        candidate_package_sha256=digest(a.candidate.read_bytes()),version=new.version_text,
        no_cnc_commands=True,verified=False,reset_sent=False)
    def save(): a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    recovery=None
    if a.recover_failed_report:
        recovery=json.loads(a.recover_failed_report.read_bytes())
        if (not a.apply or not recovery.get('preflash_disarm_and_shutter_confirmed')
                or not recovery.get('current_image_verified') or not recovery.get('error')
                or recovery.get('reset_sent') or old.image!=new.image
                or digest(new.image)!='86ee1da5da5598c643fe119e1108cadcd299a6010872fe57c5a54703970813c4'):
            raise ValueError('Recovery is restricted to verified prior 1.0.54 after a recorded failed upgrade')
        report['recovery_source']=str(a.recover_failed_report)
    try:
        if recovery is None:
            matches=[device for device in list_ports.comports()
                     if device.device.upper()==a.port.upper()
                     and device.vid==0x0483 and device.pid==0x5740
                     and device.serial_number==a.device_serial]
            if len(matches)!=1:
                raise RuntimeError('STM32 USB identity/port mismatch; refusing upgrade')
            with serial.Serial(a.port,2000000,timeout=.03,write_timeout=3) as device:
                device.dtr=True
                if (not _safe_disarm(device)
                        or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Pre-upgrade shutter/disarm confirmation missing')
                report['preflash_disarm_and_shutter_confirmed']=True
                report['preflash_all_zero_source_confirmed']=True
        session=ConnectHelper.session_with_chosen_probe(blocking=False,unique_id=a.probe,
            options={'target_override':'stm32f205re','connect_mode':'attach','auto_unlock':False,
                     'resume_on_disconnect':False,'no_config':True,'frequency':1000000,
                     'cache.enable_memory':False,'cache.enable_register':False})
        if session is None or session.probe.unique_id!=a.probe: raise RuntimeError('Wrong/missing probe')
        with session:
            t=session.target
            if recovery is None and t.get_state()!=Target.State.RUNNING: raise RuntimeError('Target not running; no automatic resume')
            words=[t.read32(0x1FFF7A10+4*i) for i in range(3)]
            swd_serial=f'{(words[0]+words[2]) & 0xFFFFFFFF:08X}{words[1]:08X}'[:12]
            if swd_serial!=a.device_serial:
                raise RuntimeError('SWD UID does not match requested USB device; refusing erase')
            report['swd_device_serial']=swd_serial
            if recovery is not None:
                report['recovery_attach_state']=str(t.get_state())
                t.reset_and_halt(Target.ResetType.SYSRESETREQ)
                if t.get_state()!=Target.State.HALTED: raise RuntimeError('Recovery reset-and-halt did not confirm')
            before=read(t,START,END-START)
            if recovery is None and before[:len(old.image)]!=old.image: raise RuntimeError('Current image differs; refusing erase')
            protected=digest(read(t,0x08000000,START-0x08000000))
            if recovery is not None and protected!=recovery['protected_sha256_before']:
                raise RuntimeError('Protected area differs from failure preflight; refusing erase')
            report['protected_sha256_before']=protected
            sectors=[]; address=START
            while address<START+len(new.image):
                region=t.memory_map.get_region_for_address(address)
                if region is None or not region.is_flash or region.flash is None: raise RuntimeError('Missing flash geometry')
                sector=region.flash.get_sector_info(address)
                if sector is None: raise RuntimeError('Missing sector geometry')
                validate_sector(sector.base_addr,sector.size)
                if not sector.base_addr<=address<sector.base_addr+sector.size: raise RuntimeError('Invalid sector geometry')
                sectors.append(dict(address=sector.base_addr,size=sector.size))
                address=sector.base_addr+sector.size
            report['erase_sectors']=sectors; report['current_image_verified']=recovery is None
            save(); print(json.dumps(report),flush=True)
            if not a.apply: return
            try:
                # Reset-and-halt stops application DMA/timers and clears exception
                # state before the SRAM flash algorithm runs. Do not run app code.
                t.reset_and_halt(Target.ResetType.SYSRESETREQ)
                report['flash_clock']=prepare_flash_clock(t)
                # Recheck access before any erase. Pack algorithms may set ACR=0;
                # do not enter them while the application PLL is still selected.
                if recovery is None and read(t,START,len(old.image))!=old.image:
                    raise RuntimeError('Readback changed after clock preparation')
                save()
                loader=MemoryLoader(session,chip_erase='sector',smart_flash=False,
                                    trust_crc=False,keep_unwritten=True)
                loader.add_data(START,new.image)
                loader.commit()
                t.halt()
                actual=read(t,START,END-START)
                if actual[:len(new.image)]!=new.image: raise RuntimeError('Application readback mismatch')
                if actual[len(new.image):]!=before[len(new.image):]: raise RuntimeError('Unwritten application tail changed')
                after=digest(read(t,0x08000000,START-0x08000000))
                if after!=protected: raise RuntimeError('Protected boot/config changed')
                report.update(verified=True,protected_sha256_after=after,
                              application_sha256=digest(new.image),unwritten_tail_preserved=True)
                save()
                t.reset()
                report['reset_sent']=True
                if t.get_state()==Target.State.HALTED:
                    pc=int(t.read_core_register('pc'))
                    if not 0x08000000<=pc<END: raise RuntimeError('Unexpected reset PC; refusing resume')
                    t.resume()
                report['state_after_reset']=str(t.get_state())
            except BaseException:
                t.halt()  # Never run an unverified/partial image.
                raise
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        print(json.dumps(report),flush=True)


if __name__=='__main__': main()
