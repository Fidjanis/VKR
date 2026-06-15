@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo [ОШИБКА] Команда "py" не найдена. Установите Python с python.org и отметьте "Add to PATH".
  echo См. файл INSTALL_WINDOWS.txt
  pause
  exit /b 1
)

echo === Версия Python ===
py -3 --version
if errorlevel 1 (
  echo [ОШИБКА] py -3 не запускается.
  pause
  exit /b 1
)

echo === pip install ===
py -3 -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ОШИБКА] pip install
  pause
  exit /b 1
)

echo === sample CSV ===
py -3 scripts\make_sample_csv.py
if errorlevel 1 (
  echo [ОШИБКА] make_sample_csv
  pause
  exit /b 1
)

echo === PPO train (может занять время) ===
py -3 scripts\train_ppo.py --csv data\sample_ohlcv.csv --timesteps 200000

echo.
echo Готово. Модель: runs\ppo_crypto\ppo_final.zip
pause
