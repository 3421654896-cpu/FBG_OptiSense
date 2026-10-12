Option Explicit

Dim shell, fso, rootDir, pythonDir, launcher, command, argument
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonDir = fso.BuildPath(rootDir, "Python")
launcher = fso.BuildPath(pythonDir, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(launcher) Then launcher = "pythonw.exe"

shell.CurrentDirectory = pythonDir
command = Quote(launcher) & " " & Quote(fso.BuildPath(pythonDir, "launch_fbg_studio.pyw")) & " --source mqtt"
For Each argument In WScript.Arguments
    command = command & " " & Quote(CStr(argument))
Next

On Error Resume Next
shell.Run command, 0, False
If Err.Number <> 0 Then
    MsgBox "FBG OptiSense Studio 启动失败：" & Err.Description, vbCritical, "FBG OptiSense Studio"
End If

Function Quote(value)
    Quote = Chr(34) & Replace(value, Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
