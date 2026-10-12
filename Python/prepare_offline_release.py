"""Developer-only signing / delivery tool. Private key never enters payloads."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from offline_update_engine import canonical_manifest


def payload(path, version, **extra):
    return dict(filename=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),version=version,**extra)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--init-key',action='store_true')
    parser.add_argument('--client-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--version',default='1.0.6')
    parser.add_argument('--firmware',type=Path)
    parser.add_argument('--device-serial',action='append',default=[])
    args=parser.parse_args()
    if args.init_key:
        if args.key.exists():parser.error('Private key exists; never replace it silently')
        args.key.parent.mkdir(parents=True,exist_ok=True)
        private=Ed25519PrivateKey.generate()
        args.key.write_bytes(private.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption()))
    else:private=Ed25519PrivateKey.from_private_bytes(args.key.read_bytes())
    public=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
    print('PUBLIC_KEY_HEX='+public,flush=True)
    if not args.output:return
    if args.output.exists():parser.error('Release directory exists; use a fresh directory')
    args.output.mkdir(parents=True)
    manifest=dict(schema='fbg_offline_update_v1',release=args.version,firmware=[])
    if args.client_dir:
        archive=args.output/f'OptiSense_Client_v{args.version}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
            for path in sorted(args.client_dir.rglob('*')):
                if path.is_file():output.write(path,path.relative_to(args.client_dir).as_posix())
        manifest['client']=payload(archive,args.version)
    if args.firmware:
        from fbg_ota_protocol import load_package
        import shutil
        fw=load_package(args.firmware)
        destination=args.output/args.firmware.name
        shutil.copy2(args.firmware,destination)
        manifest['firmware'].append(payload(destination,fw.version_text,usb_serials=args.device_serial))
    manifest['signature']=base64.b64encode(private.sign(canonical_manifest(manifest))).decode()
    (args.output/'update_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'files':len(list(args.output.iterdir()))}),flush=True)


if __name__=='__main__':main()
