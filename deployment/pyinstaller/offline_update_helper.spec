from pathlib import Path
repository = Path(SPECPATH).resolve().parents[1]
python_dir = repository/'Python'
a = Analysis([str(python_dir/'offline_update_helper.py')],pathex=[str(python_dir)],
             datas=[],binaries=[],hiddenimports=['PyQt5.sip'],hookspath=[],hooksconfig={},
             runtime_hooks=[],excludes=['PySide6','PySide2','PyQt6','tkinter','matplotlib','numpy','pyqtgraph','sklearn'],
             noarchive=False,optimize=0)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='OptiSense_Update',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=False,
        disable_windowed_traceback=False,icon=str(Path(SPECPATH)/'fbg_optisense_studio.ico'))
