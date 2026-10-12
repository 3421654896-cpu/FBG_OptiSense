# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


repository = Path(SPECPATH).resolve().parents[1]
python_dir = repository / "Python"
ota_dir = repository / "outputs" / "ota"
machine_dir = repository / "deployment" / "machine1"


def data(source, destination="."):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    return (str(source), destination)


datas = [
    data(machine_dir / "machine_profile.json"),
    data(python_dir / "finger_sensor_layout.yaml"),
    data(python_dir / "t31_touch_calibration.json"),
    data(python_dir / "fullband_equal_power_operational_2001.json"),
    data(python_dir / "fullband_equal_power_operational_2001.csv"),
    data(python_dir / "fullband_dense_high_power_2001.json"),
    data(python_dir / "fullband_equal_power_operational_2001_transition_guards.json"),
    data(python_dir / "fullband_pdt_pdr_reference_2001.json"),
    data(python_dir / "laser_calibration_checkpoint.json"),
    data(python_dir / "laser_calibration_report.json"),
    data(python_dir / "mode_tables_from_fullband_2001.json"),
    data(python_dir / "rc_dynamic_model_validation.json"),
    data(python_dir / "rc_model_accuracy_63.json"),
    data(python_dir / "stress_high_power_135_final.json"),
    data(python_dir / "stress_realtime_benchmark_latest.json"),
    data(python_dir / "stress_shape_power_dynamic_v3.json"),
    data(python_dir / "stress_shape_power_final_v2.json"),
    data(python_dir / "wave_const.yaml"),
    data(python_dir / "stress_wave_const.yaml"),
    data(python_dir / "temporary_test_default_plan.json"),
    data(machine_dir / "temporary_test_defaults.json"),
    data(
        python_dir
        / "calibration_backups"
        / "2号解调仪第一次校准"
        / "2号解调仪第一次校准_2001点.json",
        "calibration_backups/2号解调仪第一次校准",
    ),
    data(
        machine_dir
        / "client_seed"
        / "outputs"
        / "device_profiles"
        / "ota_machine_profiles.json",
        "client_seed/outputs/device_profiles",
    ),
    data(
        machine_dir / "client_seed" / "outputs" / "temporary_test" / "last_45_session.json",
        "client_seed/outputs/temporary_test",
    ),
    data(
        machine_dir / "client_seed" / "outputs" / "temporary_test" / "last_finger_capture.json",
        "client_seed/outputs/temporary_test",
    ),
    data(
        machine_dir / "client_seed" / "outputs" / "temporary_test" / "finger_captures",
        "client_seed/outputs/temporary_test/finger_captures",
    ),
    data(
        machine_dir
        / "client_seed"
        / "outputs"
        / "temporary_test"
        / "finger_captures"
        / "9个光栅最新通过.json",
        "factory_defaults",
    ),
    data(python_dir / "mach3_bridge.ps1"),
    data(python_dir / "assets", "assets"),
    data(
        python_dir / "artifacts" / "ch1_measured_template_selection_v4_certified_20260905.json",
        "artifacts",
    ),
    data(ota_dir / "JDSU_F205RE_v1.0.87.fbgfw", "firmware"),
]

for assembly_name in (
    "SolidWorks.Interop.sldworks.dll",
    "SolidWorks.Interop.swconst.dll",
):
    assembly = python_dir / assembly_name
    if assembly.is_file():
        datas.append(data(assembly))


hiddenimports = [
    "PyQt5.sip",
    "OpenGL.platform.win32",
    "matplotlib.backends.backend_agg",
]
hiddenimports += collect_submodules("sklearn")
pyocd_datas, pyocd_binaries, pyocd_hiddenimports = collect_all("pyocd")
datas.extend(pyocd_datas)
binaries = list(pyocd_binaries)
hiddenimports += pyocd_hiddenimports


a = Analysis(
    [str(python_dir / "fbg_studio_launcher.py")],
    pathex=[str(python_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide2", "PySide6", "PyQt6", "tkinter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FBG_OptiSense_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(Path(SPECPATH) / "fbg_optisense_studio.ico"),
    version=str(Path(SPECPATH) / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="app_machine1_build",
)
