# ASL-Translator

**Authors:**  
Christian (Calastian)  
Zachary

---

## Project Overview

ASL-Translator is a machine learning-powered tool that translates American Sign Language (ASL) gestures into written English text in real time. The purpose is to bridge communication gaps between Deaf/hard-of-hearing individuals and those unfamiliar with ASL, using accessible and affordable technology like webcams or smartphones.

---

## Features

- **Real-time ASL gesture recognition**
- **Text output for recognized signs**
- **User-friendly interface**
- **Modular, extensible codebase**

---


## Quick Start


1. **Unzip the folder.**
2. **Create Virtual ENV** Call it myenv: create it with python -m venv myenv, then (if linux source myenv/bin/activate) .\myenv\Scripts\activate install packages my running pip install -r (if linux linux_requirements.txt) requirements.txt
2. **Windows:** Double-click `run_windows.bat`
3. **Linux/Mac:** Open a terminal in the folder, and run:
   ```
   ./run_linux_mac.sh
   ```

The first launch will install requirements automatically.

### System Requirements

- Python 3.12.x (pre-installed on Mac/Linux; Windows users will be auto-handled)
- 8GB+ RAM recommended
- CUDA toolkit if you want GPU acceleration, or use CPU

## Model File

- The LLM model `ph-2.q4.gguf` is bundled in the `frontend/llm_model/` directory.
- The ASL_Model `ASL_MODEL_.ckpt` is bundled in the `models/` directory.

## Custom Data

- Place your video files in the appropriate folder (desktop/recordings).
- Download Phi model URL: https://huggingface.co/TheBloke/phi-2-GGUF?show_file_info=phi-2.Q4_K_M.gguf
- This is the Resource for the ASL data set: https://www.kaggle.com/datasets/abd0kamel/asl-citizen

## Troubleshooting
- For environment issues, delete the `venv` folder and re-launch the script.
