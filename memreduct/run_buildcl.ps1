$bat = "D:\code\tauri\ViewStage\memreduct\build_cl.bat"
cmd /c "call `"$bat`"" > "D:\code\tauri\ViewStage\memreduct\buildcl_out.txt" 2>&1
exit $LASTEXITCODE
