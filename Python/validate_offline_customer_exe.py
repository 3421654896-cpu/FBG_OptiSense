"""Exercise the actual frozen updater and updated client, without UI input or hardware commands.

Requires an explicitly prepared isolated installation and a fresh report directory.
The signed updater uses exactly the same UpdateWorker as its two-button GUI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updater',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    updater,target,root=args.updater.resolve(),args.target.resolve(),args.report_dir.resolve()
    expected_version = json.loads((updater.parent/'update_manifest.json').read_text(encoding='utf-8'))['client']['version']
    if root.exists():parser.error('Report directory exists; preserving previous results')
    record=target/'outputs'/'customer_record.json'
    if (not record.is_file() or json.loads(record.read_text()) .get('must_preserve') is not True
            or 'updater_cli_smoke' not in target.parts):
        parser.error('Only the explicitly prepared isolated updater_cli_smoke fixture is allowed')
    before={str(p.relative_to(target)):digest(p) for p in
            (record,target/'unins000.exe',target/'unins000.dat')}
    root.mkdir(parents=True)
    environment=dict(os.environ)
    system_root=os.environ['SystemRoot']
    environment['PATH']=system_root+'/System32;'+system_root
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','QT_QPA_PLATFORM','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):
        environment.pop(key,None)
    report=dict(physical_hardware_commands=False,external_python_path_removed=True,
                fixture_has_simulated_old_version_metadata=True)
    started=time.monotonic()
    try:
        first=root/'first_update.json'
        result=subprocess.run([str(updater),'--apply-client-update','--install-dir',str(target),
                               '--result-file',str(first)],env=environment,timeout=120,capture_output=True)
        report['updater_exit_code']=result.returncode
        report['first_update']=json.loads(first.read_text(encoding='utf-8'))
        assert result.returncode==0 and not report['first_update']['errors'],report['first_update']
        update=report['first_update']['results'][0]
        assert update['status']=='updated' and update['version']==expected_version,update
        backup=Path(update['backup'])
        report['record_and_uninstaller_retained']=all(digest(target/p)==value and digest(backup/p)==value
                                                        for p,value in before.items())
        assert report['record_and_uninstaller_retained']
        assert json.loads((target/'client_release.json').read_text())['version']==expected_version
        assert json.loads((backup/'client_release.json').read_text())['version']=='1.0.4'
        second=root/'same_version.json'
        result=subprocess.run([str(updater),'--apply-client-update','--install-dir',str(target),
                               '--result-file',str(second)],env=environment,timeout=40,capture_output=True)
        report['same_version']=json.loads(second.read_text(encoding='utf-8'))
        assert result.returncode==0 and report['same_version']['results'][0]['status']=='latest'
        environment['FBG_STUDIO_DATA_DIR']=str(root/'client_data')
        client=subprocess.Popen([str(target/'FBG_OptiSense_Studio.exe'),'--demo','--demo-seconds','8'],
                                env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            time.sleep(2)
            assert client.poll() is None,'Updated client exited before running-client gate test'
            busy=root/'running_client_gate.json'
            result=subprocess.run([str(updater),'--apply-client-update','--install-dir',str(target),
                                   '--result-file',str(busy)],env=environment,timeout=40,capture_output=True)
            report['running_client_gate']=json.loads(busy.read_text(encoding='utf-8'))
            assert result.returncode==1 and not report['running_client_gate']['results']
            assert any('关闭客户端' in message for message in report['running_client_gate']['errors'])
            _,stderr=client.communicate(timeout=60)
            report['updated_client_exit_code']=client.returncode
            assert client.returncode==0,stderr.decode(errors='replace')
        finally:
            if client.poll() is None:
                client.terminate();client.wait(timeout=15)
        report['verified']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_s']=round(time.monotonic()-started,3)
        (root/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
