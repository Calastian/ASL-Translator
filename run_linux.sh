#!/bin/bash
# Set up venv, install requirements, and run the GUI (first time only)
if [ ! -d "venv" ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install -r ./env/linux_requirements.txt
else
  source venv/bin/activate
fi
python3 -m turnIn.src.frontend.frontend_gui