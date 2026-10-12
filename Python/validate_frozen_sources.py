"""Read-only comparison of selected frozen modules against current source.

Detects a stale PyInstaller build, including edits made during Analysis.
Only compiler filename metadata is excluded from the comparison.
"""
import argparse
import marshal
from pathlib import Path
import tokenize
import types

from PyInstaller.archive.readers import CArchiveReader


def normalized(value):
    if isinstance(value, types.CodeType):
        return (
            value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount,
            value.co_nlocals, value.co_stacksize, value.co_flags, value.co_code,
            normalized(value.co_consts), value.co_names, value.co_varnames,
            value.co_name, value.co_qualname, value.co_firstlineno,
            value.co_linetable, value.co_exceptiontable, value.co_freevars,
            value.co_cellvars,
        )
    if isinstance(value, tuple):
        return tuple(normalized(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('modules', nargs='+')
    args = parser.parse_args()
    outer = CArchiveReader(str(args.exe))
    archive = outer.open_embedded_archive('PYZ.pyz')
    failed = []
    for module in args.modules:
        path = args.source_dir / (module.replace('.', '/') + '.py')
        with tokenize.open(path) as stream:
            expected = compile(stream.read(), str(path), 'exec', optimize=0, dont_inherit=True)
        try:
            actual = archive.extract(module)
        except KeyError:
            # A PyInstaller entry script lives in the outer CArchive.
            actual = marshal.loads(outer.extract(module))
        matches = normalized(expected) == normalized(actual)
        print(f'{module}: {"MATCH" if matches else "STALE"}', flush=True)
        if not matches:
            failed.append(module)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
