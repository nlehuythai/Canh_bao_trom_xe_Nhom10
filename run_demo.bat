@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m pip install opencv-python numpy
python main.py --demo --output-dir outputs
pause
