# OptiSense 最新源码开发交接（2026-10-10）

本包是当前本地工作目录快照，包含未提交/未跟踪的新源码，不是 GitHub 老版本导出。
客户端 v1.0.7，STM32F205RE 应用固件源码 v1.0.110。
这是打包脚本使用的开发交接说明模板；修改模板不会更新此前生成的 v1.0.109 ZIP。
MANIFEST.sha256.json 记录 Git 基线、每个文件 SHA-256、来源和脱敏字段。

## 包含范围

- Python：客户端、USB/LAN、自动校准/选点、密集光谱、手指应力、温度/风扇协议、离线更新助手及全部当前顶层 Python 工具和单元测试。
- JDSU：应用、USB CDC、HAL/CMSIS、中间件、Bootloader、CubeMX/Keil 工程和散装加载配置。
- deployment：PyInstaller/Inno Setup 配置、打包脚本、已筛选的通用初始化数据。
- machine_defaults、现存运行/分析/测试 JSON/CSV/YAML 数据；保留原有版本和标注，不代表全部验收通过。
- backups：原 45 点参考及隐藏旧功能的可编辑源码；不含旧二进制/板卡私有备份。
- outputs/ota：v1.0.87/106/109 是现有构建配置兼容资源，不是最新源码版本或推荐降级版本；v1.0.110 镜像需从当前源码重新构建并验证。

## 环境和安全启动

Windows 10/11 x64、CPython 3.12 x64（本地 3.12.14）。在解压根目录执行：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_dev.ps1
.\start_demo.ps1
~~~

setup_dev.ps1 在 Python\.venv 安装 requirements-lock.txt 精确依赖，首次需联网。
演示模式不连接/控制硬件。IDE 解释器为 Python\.venv\Scripts\python.exe，入口 Python\fbg_studio_launcher.py。
确认设备身份、光路、电流上限和停止手段后，才执行 start_dev.ps1。
串口需 VCP/CH343 驱动；AQ6150B 需 VISA Runtime；Mach3/SolidWorks 仅相关研发功能需要。
现场 Wi-Fi/OTA/MQTT 密码和 DPAPI 密文不随包，开发者须自行配置。
默认机号不代表实际硬件；禁止凭历史 COM/IP 值烧录。

## 最新行为与不能回退的要求

- 全部机号/后续固件统一 40℃ 温控。正常校准无满转覆盖；启动/传感器/PWM 故障安全全速保留。详见 JDSU/AGENTS.md。
- 正式固件关闭旧应力/温度寻峰、MQTT 广域网、网络 OTA；保留 USB 更新、USB/LAN 数据、共享 EXTRA 和精确安全 DISARM ACK。
- 客户端隐藏内置更新页/在线提示；采用发送完整包、客户运行独立助手的离线交付。
- 手指测试默认连续循环；启动峰内/跨峰等待均 850 µs；参考线和实时样条拟合开关常显；移除客户“未通过点（红×）”按钮和部分说明文字。
- 45 点参考每点等 10 ms 后采一次新鲜原始 ADC，无光学稳定性/峰形筛除或补采。保留身份、协议、电流边界和安全关光检查。0.45 s 只是等待预算，不等于整次实测耗时。
- 二/三号机 v1.0.109 曾通过协议/零电流功能检查；其 CH0 非 2 kΩ 的临时流随后发现 0200 拒绝并在 v1.0.110 修复。
- 三号机已烧录 v1.0.110，全部 10 种通道/挡位组合各 32 帧零电流协议实测通过；二号机不能由此推定已升级。三号机临时 2001 点数据仍非全部光学验收通过，不能改称正式完成。

## 无硬件回归

~~~powershell
Push-Location Python
.\.venv\Scripts\python.exe -m unittest test_thermal_fan_policy test_firmware_feature_profile test_firmware_scan_scheduler test_temporary_feedback_policy
.\.venv\Scripts\python.exe -m unittest test_temporary_reference test_dense_scan_transport test_temporary_test_widget test_fbg_unified_app test_ui_responsive_layout test_requested_client_changes test_offline_update_engine test_offline_update_helper
Pop-Location
~~~

不要用 live/capture/calibrate/flash 工具替代无硬件单元测试：这些工具可能开光、驱动机床或写设备。

## 构建与正式更新

固件需合法安装 Keil MDK-ARM 5 / ARM Compiler 6，工程 JDSU\MDK-ARM\JDSU.uvprojx。
Bootloader\build_ota_release.ps1 默认 Keil 路径 D:\Keil_v5，需先核实/调整。
明确传版本参数（固件脚本默认 1.0.0，不要省略）：

~~~powershell
.\JDSU\Bootloader\build_ota_release.ps1 -Version 1.0.110
.\Python\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\deployment\build_universal_update.ps1 -Version 1.0.7
~~~

客户端另需 Inno Setup 6。中文 venv 路径在打包时使用脚本内 ASCII 目录联接。
离线助手配置：deployment\pyinstaller\offline_update_helper.spec；流程：deployment\offline_update_workflow.txt。
改动后增加版本，回归并实机验收；编译通过不等于安装/光学验收通过。
更新使用 Ed25519 清单签名和载荷 SHA-256；Python/update_public_key.py 只包含公钥。
生产私钥由发布者保存，不随包。开发者可构建/测试；现有客户接受的正式包须由持钥者签名。
另一套新密钥不会自动被已交付客户信任，禁止覆盖公钥后当成兼容更新发布。

## 排除与保密

不含 Git 历史、venv、EXE/客户安装器、缓存、私有全 Flash、现场输出日志、签名私钥和真实凭据。
大型 joblib 模型需用授权数据和训练/校准工具重建；主客户端启动不依赖。
第三方编译工具和 BW20 厂商源码不随包。
包内含内部源码和标定数据，仅给受信任开发者，不用于客户交付或直接公开发布。

## 校验/再打包

~~~powershell
.\deployment\verify_developer_source_bundle.ps1 -ZipPath "源码包.zip"
.\deployment\build_developer_source_bundle.ps1
~~~

打包器只读工作文件，不控制硬件、不读私钥、不覆盖已有 ZIP。
验证 ZIP CRC、逐文件 SHA-256、Python 语法、关键源码和敏感 JSON 字段。
