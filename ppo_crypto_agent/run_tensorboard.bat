@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Запуск TensorBoard (остановка: Ctrl+C)
echo Логи: %cd%\runs\ppo_crypto\tb
py -3 -m tensorboard.main --logdir runs\ppo_crypto\tb
pause
