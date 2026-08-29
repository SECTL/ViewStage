@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
if errorlevel 1 ( echo [VCVAR_FAIL] & exit /b 1 )

set ROOT=D:\code\tauri\ViewStage\memreduct
set OUT=%ROOT%\bin\64
if not exist "%OUT%" mkdir "%OUT%"
set OBJ=%ROOT%\obj64
if not exist "%OBJ%" mkdir "%OBJ%"

set CFLAGS=/nologo /c /O2 /Ob2 /Oi /Oy /GF /Gy /Gz /MT /W3 /Zi /guard:cf /permissive- ^
 /DMICROSOFT_WINDOWS_WINBASE_H_DEFINE_INTERLOCKED_CPLUSPLUS_OVERLOADS ^
 /D_UNICODE /DUNICODE /DWIN32_LEAN_AND_MEAN ^
 /DAPP_HAVE_AUTORUN /DAPP_HAVE_SKIPUAC /DAPP_HAVE_TRAY /DAPP_HAVE_UPDATES /DNDEBUG
set INCS=/I "%ROOT%\routine\src" /I "%ROOT%\src\include" /I "%ROOT%\src"

echo [1/3] compiling C sources...
cl %CFLAGS% %INCS% "%ROOT%\routine\src\rapp.c" /Fo"%OBJ%\rapp.obj"
if errorlevel 1 ( echo [CL_FAIL rapp] & exit /b 1 )
cl %CFLAGS% %INCS% "%ROOT%\routine\src\routine.c" /Fo"%OBJ%\routine.obj"
if errorlevel 1 ( echo [CL_FAIL routine] & exit /b 1 )
cl %CFLAGS% %INCS% "%ROOT%\src\main.c" /Fo"%OBJ%\main.obj"
if errorlevel 1 ( echo [CL_FAIL main] & exit /b 1 )

echo [2/3] compiling resources...
rc /nologo /D_UNICODE /DUNICODE /D_WIN64 /I"%ROOT%\src" /Fo"%OBJ%\resource.res" "%ROOT%\src\resource.rc"
if errorlevel 1 ( echo [RC_FAIL] & exit /b 1 )

echo [3/3] linking...
set LFLAGS=/nologo /SUBSYSTEM:WINDOWS /DEBUG /OPT:REF /OPT:ICF /RELEASE /GUARD:CF /CETCOMPAT /DEPENDENTLOADFLAG:0x800 /OUT:"%OUT%\memreduct-viewstage.exe" /PDB:"%OUT%\memreduct-viewstage.pdb"
set LIBS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib psapi.lib taskschd.lib comctl32.lib shlwapi.lib version.lib rpcrt4.lib gdiplus.lib
link %LFLAGS% "%OBJ%\rapp.obj" "%OBJ%\routine.obj" "%OBJ%\main.obj" "%OBJ%\resource.res" %LIBS%
if errorlevel 1 ( echo [LINK_FAIL] & exit /b 1 )

echo [BUILD_OK] %OUT%\memreduct-viewstage.exe
exit /b 0
