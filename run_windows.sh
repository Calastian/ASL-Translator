@echo off
REM Set up venv, install requirements, and run the GUI (first time only)
IF NOT EXIST venv (
    python -m venv venv
    call venv\Scripts\activate
    pip install -r .\env\requirements.txt
) ELSE (
    call venv\Scripts\activate
)
REM Run your frontend app
python -m turnIn.src.frontend.frontend_gui
pause