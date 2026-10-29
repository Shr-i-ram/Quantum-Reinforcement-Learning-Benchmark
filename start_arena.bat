@echo off
echo ======================================================================
echo    Starting Quantum RL Arena - Interactive Research Laboratory
echo ======================================================================

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Launch Python runner
python run_arena.py
pause
