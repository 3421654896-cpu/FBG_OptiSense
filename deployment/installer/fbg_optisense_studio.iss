#define MyAppName "FBG OptiSense Studio"
#define MyAppNameZh "光栅解调工作站"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "OptiSense Studio"
#define MyAppExeName "FBG_OptiSense_Studio.exe"

[Setup]
AppId={{7E127A4E-06B6-4B82-A632-CF8C8443CDA8}
AppName={#MyAppNameZh}
AppVersion={#MyAppVersion}
AppVerName={#MyAppNameZh} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\FBG OptiSense Studio
DefaultGroupName={#MyAppNameZh}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\Python\installer
OutputBaseFilename=FBG_OptiSense_Studio_Machine1_v{#MyAppVersion}_Setup
SetupIconFile=..\pyinstaller\fbg_optisense_studio.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
CloseApplications=yes
RestartApplications=no
MinVersion=10.0
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppNameZh} 一号机安装程序
VersionInfoProductName={#MyAppNameZh}
VersionInfoProductVersion={#MyAppVersion}

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "..\..\Python\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Customer records and encrypted machine settings live outside {app} in
; LocalAppData\FBGOptiSenseStudio and therefore survive an in-place update.
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\{#MyAppExeName}"

[Icons]
Name: "{group}\{#MyAppNameZh}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\卸载 {#MyAppNameZh}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppNameZh}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppNameZh}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
