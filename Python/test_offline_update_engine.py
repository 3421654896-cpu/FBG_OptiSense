import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from offline_update_engine import *


class OfflineUpdateTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.target=self.root/'OptiSense'
        self.target.mkdir();(self.target/CLIENT_EXE).write_bytes(b'old')
        (self.target/'_internal').mkdir()
        self.zip=self.root/'client.zip'
        with zipfile.ZipFile(self.zip,'w') as archive:
            archive.writestr(CLIENT_EXE,b'new');archive.writestr('_internal/runtime.dll',b'runtime')

    def test_signature_valid_and_tampering_rejected(self):
        key=Ed25519PrivateKey.generate()
        public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
        manifest={'schema':'fbg_offline_update_v1','release':'1.0.6'}
        manifest['signature']=base64.b64encode(key.sign(canonical_manifest(manifest))).decode()
        file=self.root/'update_manifest.json';file.write_text(json.dumps(manifest))
        self.assertEqual(load_manifest(file,public),manifest)
        manifest['release']='9.9.9';file.write_text(json.dumps(manifest))
        with self.assertRaises(UpdateError):load_manifest(file,public)

    def test_hash_gate(self):
        item={'filename':self.zip.name,'bytes':self.zip.stat().st_size,
              'sha256':hashlib.sha256(self.zip.read_bytes()).hexdigest()}
        self.assertEqual(checked_payload(self.root,item),self.zip.resolve())
        self.zip.write_bytes(b'corrupt')
        with self.assertRaises(UpdateError):checked_payload(self.root,item)
        item['filename']='../x'
        with self.assertRaises(UpdateError):checked_payload(self.root,item)

    def test_device_isolation(self):
        manifest={'firmware':[{'usb_serials':['395736523033'],'version':'1.0.109'}]}
        self.assertEqual(firmware_for_serial(manifest,'395736523033')['version'],'1.0.109')
        with self.assertRaises(UpdateError):firmware_for_serial(manifest,'other')

    def test_replace_preserves_user_data_and_backup(self):
        record=self.target/'outputs'/'user.json';record.parent.mkdir();record.write_text('keep')
        outside=self.root/'LocalAppData';outside.mkdir();(outside/'calibration.json').write_text('keep')
        result=replace_client(self.zip,self.target,'1.0.6')
        self.assertEqual((self.target/CLIENT_EXE).read_bytes(),b'new')
        self.assertEqual(record.read_text(),'keep')
        self.assertEqual((Path(result['backup'])/CLIENT_EXE).read_bytes(),b'old')
        self.assertEqual((outside/'calibration.json').read_text(),'keep')
        self.assertEqual(replace_client(self.zip,self.target,'1.0.6')['status'],'latest')
        self.assertEqual(replace_client(self.zip,self.target,'1.0.5')['status'],'latest')

    def test_zip_traversal_does_not_modify_original(self):
        with zipfile.ZipFile(self.zip,'w') as archive:archive.writestr('../outside.txt',b'bad')
        with self.assertRaises(UpdateError):replace_client(self.zip,self.target,'1.0.6')
        self.assertEqual((self.target/CLIENT_EXE).read_bytes(),b'old')
        self.assertFalse((self.root/'outside.txt').exists())

    def test_partial_install_rename_failure_rolls_back(self):
        rename=Path.rename
        def fail_stage(path,dest):
            if '.staging-' in path.name:raise OSError('injected failure')
            return rename(path,dest)
        with patch.object(Path,'rename',fail_stage):
            with self.assertRaises(OSError):replace_client(self.zip,self.target,'1.0.6')
        self.assertEqual((self.target/CLIENT_EXE).read_bytes(),b'old')
        self.assertFalse(list(self.root.glob('*.staging-*')))

    def test_broad_directory_rejected(self):
        for target in (Path.home(),Path(self.root.anchor),self.root/'missing'):
            with self.assertRaises(UpdateError):validate_install_target(target)

    def test_busy_client_not_killed(self):
        class Process:
            info={'exe':str(self.target/CLIENT_EXE)}
        with patch('psutil.process_iter',return_value=[Process()]):
            with self.assertRaisesRegex(UpdateError,'关闭客户端'):
                replace_client(self.zip,self.target,'1.0.6')
        self.assertEqual((self.target/CLIENT_EXE).read_bytes(),b'old')

    def test_installer_uninstaller_is_preserved(self):
        for name in ('unins000.exe','unins000.dat','unins001.msg'):
            (self.target/name).write_bytes(b'keep-uninstaller')
        result=replace_client(self.zip,self.target,'1.0.6')
        for name in ('unins000.exe','unins000.dat','unins001.msg'):
            self.assertEqual((self.target/name).read_bytes(),b'keep-uninstaller')
            self.assertEqual((Path(result['backup'])/name).read_bytes(),b'keep-uninstaller')

    def test_windows_user_data_junction_is_rejected(self):
        linked=self.target/'outputs';linked.mkdir()
        with patch.object(Path,'is_junction',create=True,
                          side_effect=lambda:False):
            # Patch only the link predicate; no real junction or outside writes.
            original=__import__('offline_update_engine')._is_link
            with patch('offline_update_engine._is_link',side_effect=lambda p:p.resolve()==linked.resolve() or original(p)):
                with self.assertRaisesRegex(UpdateError,'用户数据包含链接'):
                    replace_client(self.zip,self.target,'1.0.6')
        self.assertEqual((self.target/CLIENT_EXE).read_bytes(),b'old')
