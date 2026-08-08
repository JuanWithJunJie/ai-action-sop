@echo off
REM AI-SOP Vision - Windows 打包脚本（PyInstaller）
REM 用法：双击运行，或在命令行执行 build.bat
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] 未找到 .venv，请先初始化环境:
    echo   py -3.10 -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

echo [1/3] 安装 pyinstaller ...
.venv\Scripts\pip install pyinstaller || exit /b 1

echo [2/3] 清理旧构建 ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/3] 开始打包（首次打包需几分钟）...
.venv\Scripts\pyinstaller --noconfirm --clean --windowed --name AI-SOP-Vision ^
    --collect-data mediapipe ^
    --add-data "lstm_runs_fine;lstm_runs_fine" ^
    --add-data "config.json;." ^
    ai_sop_gui.py || exit /b 1

echo.
echo 打包完成: dist\AI-SOP-Vision\AI-SOP-Vision.exe
echo 分发时请连同 dist\AI-SOP-Vision\ 整个目录一起拷贝。
endlocal
