# 客户端与解调仪固件的分开发布

> 历史研发流程，当前客户版本不默认使用。自 2026-10-10 起，主软件隐藏更新页和气泡并停止自动 GitHub 检查；客户使用开发者发送的完整签名包及独立更新助手。网络 OTA 在正式固件中关闭，USB 更新保留。以下内容仅供显式恢复该研发功能时参考，不能描述为当前客户行为。当前流程见 `offline_update_workflow.txt` 和仓库主 README。

客户程序从公开的 `3421654896-cpu/JDSU_Laser` GitHub Releases 检查更新：启动后检查一次，此后每 6 小时检查一次。优先读取最新正式 Release 内的 `FBG_OptiSense_updates.json` 清单；旧 Release 没有清单时才回退到 GitHub API（匿名 API 可能被限流）。检查本身不安装，也不控制激光器。客户在侧栏看到独立的“客户端有新版本”和“固件有新版本”气泡，进入对应页面分别点一次更新。

## 客户端安装包

1. 更新 `Python/app_version.py`、`deployment/installer/fbg_optisense_studio_universal.iss` 和 Windows 版本资源中的版本号。
2. 按现有 `deployment/build_universal_update.ps1` 构建并在本机验证新安装包。不要把开发目录里的临时数据当成客户数据打包。
3. 创建非草稿、非预发布的公开 GitHub Release，上传名称严格为 `FBG_OptiSense_Studio_Universal_vX.Y.Z_Setup.exe` 的安装包。例如 `FBG_OptiSense_Studio_Universal_v1.0.5_Setup.exe`。更新清单或 GitHub Release API 必须为该资源提供正确的 `sha256:...` 摘要；没有摘要的软件包不会出现在一键更新中。
4. 用户点“客户端更新”后，程序从 GitHub 下载并核验完整 SHA-256。正式安装版会先安全关闭软件（包括未保存光谱提示），再启动现有 Inno Setup 安装程序原位覆盖；源码开发版只下载并提示手动运行，不会覆盖源码。

## STM32 固件包

1. 每台解调仪的校准点表都可能不同。将已通过该机验收的 `.fbgfw` 固件包单独上传，名称严格为 `JDSU_F205RE_machine_N_vX.Y.Z.fbgfw`，例如 `JDSU_F205RE_machine_3_v1.0.97.fbgfw`。**不能仅把另一台机的固件改名后发布。** 文件名版本须与包头内的固件版本一致。
2. 客户程序只下载当前所选机号对应的公开 Release 资源，先核验 GitHub SHA-256，再核验 `.fbgfw` 包头、整包 CRC、STM32 向量表与版本。没有明确机号的通用固件不会自动提供给客户。
3. 程序从已收到的设备状态帧读取当前固件版本。发布包已就绪但还没有设备当前版本时，只提示“发现固件发布、需要连接设备确认”；只有发布版本严格更高、该机的局域网 IP 与 OTA 密码已配置，才启用“一键更新”。升级仍保留现有的写入、整包校验、重启回执和 RAW READY 验证。没有当前版本状态或无法确认包与机号时不会自动刷写。
4. 首次具备 OTA 能力仍需先用调试器写入 OTA 引导程序。每台设备在客户所用的 Windows 用户下还需由开发者完成一次 IP/OTA 授权配置；密码按 Windows 用户加密保存，不能把明文密码写进公开 Release 或安装包。源码开发版可通过“端口与设备选项 → OTA 开发者设置”进入；正式安装版需由开发者设置环境变量 `FBG_STUDIO_DEVELOPER_MODE=1` 后重启程序才显示这个入口。客户默认只看到更新按钮和状态。

## 每次发布的更新清单

在 `Python` 目录运行 `build_release_update_manifest.py`，例如：

```powershell
python build_release_update_manifest.py --tag v1.0.5 --asset FBG_OptiSense_Studio_Universal_v1.0.5_Setup.exe --asset JDSU_F205RE_machine_3_v1.0.97.fbgfw --out FBG_OptiSense_updates.json
```

脚本读取本地文件，计算长度和 SHA-256，并检查命名和机号。把生成的 `FBG_OptiSense_updates.json` 与对应二进制文件一起上传到同一个公开正式 Release。若本次只更新客户端，但还要让旧版固件继续可发现，可用 `--asset-tag 旧Release标签 旧固件文件路径` 将旧资源一同列入新清单；旧文件必须仍在原公开 Release 中，清单中的摘要也要与旧文件一致。客户端和固件可以独立升级，但**最新 Release 的清单必须列出目前仍要提供的每个更新资源**。无清单时 API 回退可扫描最近的公开 Releases，但可能遇到匿名访问限流。

第一次让现有客户拥有这个功能，仍需要先发布并发给客户一次包含新版更新程序的安装包；旧安装版不会凭空出现新按钮。公开仓库没有 Release、网络断开或摘要缺失时只显示不可更新状态，不会静默降级、使用未核验文件或尝试刷写。
