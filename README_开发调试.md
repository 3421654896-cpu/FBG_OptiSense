# FBG OptiSense Studio 多机通用版开发调试源码包

本包是桌面解调程序 **v1.0.4 多机通用版** 的最小可运行/可调试源码包。它包含主程序实际使用的本地 Python 模块、一号机和二号机各自的 2001 点数据、分机临时模式与局域网/OTA 配置、OTA 1.0.87 固件包，以及固定版本的 Python 依赖清单。

## 环境要求

- Windows 10/11 x64
- CPython 3.12 x64（制作和验证本包时使用 3.12.14）
- 联机调试时按硬件需要安装 STM32 VCP/CH343 串口驱动
- 使用 AQ6150B 时安装 NI-VISA 或 Keysight VISA Runtime
- 只有使用机床功能时才需要安装并配置 Mach3
- 只有使用 SolidWorks 联动时才需要本机安装 SolidWorks

## 首次安装

在 PowerShell 中进入本目录后执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_dev.ps1
```

脚本会创建独立的 `.venv` 并严格按 `requirements-lock.txt` 安装依赖，不会修改系统 Python。依赖需要从 PyPI 下载，因此首次安装需要联网。

## 启动

安全演示模式（不连接/控制硬件）：

```powershell
.\start_demo.ps1
```

真实设备开发模式：

```powershell
.\start_dev.ps1
```

`start_dev.ps1` 会连接真实设备并可能控制激光器；执行前请确认光路、电流上限、急停和设备编号。

也可以在 IDE 中将解释器设为 `.venv\Scripts\python.exe`，入口设为 `Python\fbg_unified_app.py`。

## 多机内容

程序默认选择一号机，可在软件左下角切换机号。切换后会同时切换 2001 点数据、临时模式记录和 OTA 参数，不共用现场采集记录。

- 一号机：`192.168.3.46`，正式 2001 点（2001/2001），带 9 峰/45 点默认记录
- 二号机：`192.168.3.26`，第一次校准数据（1716/2001，285 点待复测），9 峰/3 峰路线为空，需在二号机实物上采集
- 设备注册表：`Python\machine_registry.json`
- 分机临时模式默认值：`Python\machine_defaults\<machine_id>\temporary_test_defaults.json`
- OTA 固件：`Python\firmware\JDSU_F205RE_v1.0.87.fbgfw`

开发时新增解调仪，应先在 `machine_registry.json` 登记机号和校准文件，再增加对应的 `machine_defaults` 与 OTA 配置；不要复制出新的机号专版主程序。

## 安全与隐私

本包不含 OTA 密码、DPAPI 密文、现场日志或未筛选的历史输出。随包的一号机临时模式记录仅是正式通用版需要的已筛选默认种子。首次在另一台电脑使用 OTA 时，需要按机号重新输入密码。MQTT 用户名和密码应通过环境变量 `FBG_MQTT_USERNAME`、`FBG_MQTT_PASSWORD` 提供。

## 有意排除的内容

- `.venv`、第三方库二进制副本、PyInstaller 构建缓存、EXE/安装包、测试缓存和历史备份
- 现场 `outputs` 日志和未筛选的实验数据
- 约 330 MiB 的 `fullband_equal_power_surrogate.joblib`（校准训练产物，可由校准工具重新生成，主程序运行不依赖）
- STM32/BW20 完整固件工程和编译工具链

桌面程序可以使用本包内现成的 OTA 固件进行 OTA 调试；若要修改并重新编译板卡固件、重建模式表固件，需要完整的原始固件仓库和对应工具链。

第三方 Python 库没有整包复制进 ZIP，而是以精确版本记录在 `requirements-lock.txt`，由 `setup_dev.ps1` 安装到包内独立 `.venv`。这样既能复现环境，也避免把数百 MiB、特定 Python 小版本绑定的二进制文件混入最小源码包。

## 文件校验

`MANIFEST.sha256.json` 记录了包内有效载荷文件的 SHA-256 和字节数。ZIP 外侧另有 `.sha256` 文件，用于传输后校验整个压缩包。
