#!/bin/bash

.\myenv\Scripts\activate

# Run your frontend app
echo "Starting ASL Translator GUI..."
python -m turnIn.src.frontend.frontend_gui
