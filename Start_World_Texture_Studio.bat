@echo off
chcp 65001 >nul
title Eagle World Texture & Landmark Studio (Port 8950)
cd /d "%~dp0"
echo ============================================================================
echo   EAGLE WORLD TEXTURE ^& LANDMARK STUDIO — WIBU WORLD BUILDER
echo   Target Game: C:\Users\LENOVO\Desktop\GTA\Project Eagle\gta sa gameplay-lab
echo   URL:         http://127.0.0.1:8950
echo ============================================================================
start "" "http://127.0.0.1:8950"
python app.py
pause
