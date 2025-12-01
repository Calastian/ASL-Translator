#!/bin/bash
# filepath: /home/calastian/Documents/AppState/Capstone/ASL-Translator/run_linux.sh

# Set up venv, install requirements, and run the GUI (first time only)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install -r ./env/requirements.txt
else
    echo "Activating existing virtual environment..."
    source venv/bin/activate
fi

# Run your frontend app
echo "Starting ASL Translator GUI..."
python -m turnIn.src.frontend.frontend_gui

# Keep terminal open (bash equivalent of pause)
read -p "Press Enter to exit..."