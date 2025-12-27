# Pizza-Store-Scooper-Violation-Detection

<!--
Place your screenshot file at `docs/overview.png`. If you prefer another path,
update the path above. Example: `assets/architecture.png`.
-->

A computer vision pipeline for detecting scooper hygiene violations in a pizza store.  
Built with GStreamer (frame reader), FastAPI (API), PostgreSQL (database), and React.js (frontend).

---

## 🎥 Demo Video
[![Watch the demo](demo-thumbnail.png)](https://github.com/Mazen-Ahmed12/Pizza-Store-Scooper-Violation-Detection/releases/tag/v1.0.0/demo.mp4)

---
## Architecture

Code:  
`Camera (RTSP) ➜ GStreamer Frame Reader ➜ Detection Service ➜ FastAPI API ➜ PostgreSQL ➜ Frontend (React.js)`

- Frame Reader (GStreamer): Decodes RTSP/video files with minimal FPS drop.
- Detection Service: Runs YOLO-based detection, applies ROI logic for scooper hygiene.
- API (FastAPI): Exposes endpoints (`/violations`, `/stream`) for real-time data.
- Database (PostgreSQL): Stores structured violation logs (frame ID, timestamp, bounding boxes, labels).
- Frontend (React.js): Displays live stream and violation history.

---

## Tech Choices

| Service      | Choice       | Why Chosen                                              | Why Not Alternatives                                    |
|--------------|--------------|---------------------------------------------------------|---------------------------------------------------------|
| Frame Reader | GStreamer    | Real-time RTSP decoding, minimal FPS drop, easy integration | OpenCV drops frames; FFmpeg harder to integrate; DeepStream GPU-only & complex |
| Database     | PostgreSQL   | Strong schema, relational queries, recruiter-friendly    | MongoDB flexible but weaker for structured logs         |
| API          | FastAPI      | Async, fast, auto docs, scalable microservices          | Flask slower, sync-only, manual validation/docs         |
| Frontend     | React.js     | Flexible, modern, recruiter-friendly, matches skillset  | Vue.js less familiar; Streamlit/Gradio limited flexibility |

---

### Top API endpoints
- GET /stream → Live annotated video stream (frames with bounding boxes and ROI overlays).
- GET /violations → List of violation logs stored in PostgreSQL (frame ID, timestamp, bounding boxes, labels, violation status).
- POST /detect → Run detection on an uploaded frame or video snippet and return violation metadata.
- GET /health → Service health check to verify API and microservice connectivity.
- GET /stats → Analytics summary (e.g., violations per hour/day, most common violation types).
- POST /config → Update ROI or detection parameters (e.g., scooper region, cooldown settings) at runtime.

---

## Table of Contents
- [Requirements](#requirements)
- [Quick start (CPU or existing CUDA)](#quick-start-cpu-or-existing-cuda)
- [CUDA installation (choose version for your GPU)](#cuda-installation-choose-version-for-your-gpu)
  - [1. Check your GPU and driver](#1-check-your-gpu-and-driver)
  - [2. Decide which CUDA version to install](#2-decide-which-cuda-version-to-install)
  - [3. Install NVIDIA driver (Linux)](#3-install-nvidia-driver-linux)
  - [4. Install CUDA Toolkit (Linux / Windows)](#4-install-cuda-toolkit-linux--windows)
  - [5. Install matching PyTorch / TensorFlow wheel](#5-install-matching-pytorch--tensorflow-wheel)
  - [Verification](#verification)
  - [Notes and troubleshooting](#notes-and-troubleshooting)
- [Environment setup (virtualenv / conda)](#environment-setup-virtualenv--conda)
- [Install Python dependencies](#install-python-dependencies)
- [Launch (backend & frontend)](#launch)
- [Acknowledgements](#acknowledgements)

---

## Requirements
- Python 3.10 (recommended) or 3.11 
- git
- (Optional but recommended for speed) NVIDIA GPU + compatible driver and CUDA/cuDNN for GPU acceleration
- Internet connection to download packages and models
- Node.js + npm (for the frontend)

---

## Quick start (CPU or existing CUDA)
1. Clone the repo:
   ```
   git clone https://github.com/Mazen-Ahmed12/Pizza-Store-Scooper-Violation-Detection.git
   cd Pizza-Store-Scooper-Violation-Detection
   ```
2. Create and activate a Python environment (see below).
3. Install Python dependencies: 
   ```
   pip install -r requirements.txt
   ```
### Notes
- If you have an NVIDIA GPU, install CUDA and the matching PyTorch wheel before installing other packages.
  - Some packages (for example `ultralytics`) check for an existing `torch` and may auto-install a default/CPU-only wheel if none is present.

4. Run inference/training (examples below).

If you already have a functioning CUDA setup and a GPU-enabled framework installed (for example PyTorch with CUDA), you can skip straight to the virtualenv/requirements and run scripts.

---

## CUDA installation (choose version for your GPU)

This section explains how to install CUDA tooling and matching Python packages. IMPORTANT: pick a CUDA version that is compatible with:
- your GPU compute capability and drivers
- the deep learning framework version you will install (PyTorch / TensorFlow)

If you prefer not to manage CUDA yourself, use a pre-built Docker image or a cloud instance with CUDA preinstalled.

### 1. Check your GPU and driver
- Show GPU and driver:
  ```
  nvidia-smi
  ```
  Output shows GPU model and driver version. Note the driver version — some CUDA toolkits require a minimum driver version.
- If `nvidia-smi` is not available, there may be no NVIDIA driver installed.

### 2. Decide which CUDA version to install
- Check the compatibility matrix for your chosen framework:
  - PyTorch: https://pytorch.org/get-started/locally/
  - TensorFlow: https://www.tensorflow.org/install/source#gpu
- Check NVIDIA CUDA Toolkit downloads and compatibility: https://developer.nvidia.com/cuda-toolkit
- Choose a CUDA Toolkit version that:
  - Is supported by your GPU (older GPUs may not support very new CUDA)
  - Has wheels available for the deep learning framework version you plan to install

If unsure, a safe workflow is:
- Install the latest recommended NVIDIA driver for your GPU
- Install a widely-used CUDA version (e.g., CUDA 11.x or 12.x) that your framework supports
- Install PyTorch/TensorFlow using the official command for that CUDA version

### 3. Install NVIDIA driver (Linux example)

Windows: Use the NVIDIA Driver download page or GeForce Experience to install the appropriate driver for your GPU.

On Ubuntu you can use the driver manager:
```
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```
After reboot, verify with `nvidia-smi`.


### 4. Install CUDA Toolkit (Linux / Windows)
Follow NVIDIA's official instructions for your OS and chosen CUDA version:

- NVIDIA CUDA Toolkit downloads: https://developer.nvidia.com/cuda-toolkit
- Choose the OS, architecture, distribution and installer type, then follow the step-by-step instructions given on NVIDIA’s site.

 option to avoid driver conflicts.

### 5. Install matching PyTorch / TensorFlow wheel
After installing CUDA and drivers, install the Python packages that were built for that CUDA version.

Examples for PyTorch (adjust for the CUDA version you installed):
- CUDA 12.1:
  ```
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
- CUDA 11.8:
  ```
  pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
  ```
- CPU-only:
  ```
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  ```

For TensorFlow, use the pip wheel matching your CUDA combination (see TensorFlow install docs).

If using conda, prefer `conda install` channels for pinned compatibility:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Replace `11.8` with your chosen CUDA version.

### Verification
- Verify driver and GPU:
  ```
  nvidia-smi
  ```
- Verify nvcc (CUDA compiler):
  ```
  nvcc --version
  ```
- Verify in Python (example for PyTorch):
  ```py
  python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
  ```

### Notes and troubleshooting
- If `torch.cuda.is_available()` is false but `nvidia-smi` shows GPUs, check:
  - Driver compatibility with CUDA version
  - Installed PyTorch wheel matches the installed CUDA
  - LD_LIBRARY_PATH or PATH include CUDA and cuDNN library locations
- When in doubt, use conda environments and conda packages from `nvidia` and `pytorch` channels to reduce mismatch issues.
- Consider using Docker images with CUDA preinstalled (NVIDIA NGC, official PyTorch images) to avoid host-level installation issues.

---

## Environment setup (virtualenv / conda)

Virtualenv (recommended for minimal systems):
```
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install --upgrade pip
```

Conda:
```
conda create -n pizza-det python=3.10
conda activate pizza-det
```

---

## Install Python dependencies
After activating your environment:
```
pip install -r requirements.txt
```
If your framework (PyTorch/TensorFlow) requires a specific CUDA build, install that first (see section above), then re-run `pip install -r requirements.txt` or install the framework with the CUDA-specific wheel before installing other requirements.

---

## launch 

Run the backend server:
```
cd fast_api
start uvicorn fast_api.api:app --reload --port 8000
```

- you ensure that the server run when you see this in cmd(or whatever you use)
```
Uvicorn running on ←[1mhttp://127.0.0.1:8000←[0m (Press CTRL+C to quit)
←[32mINFO←[0m:     Started reloader process [←[36m←[1m29624←[0m] using ←[36m←[1mWatchFiles←[0m
←[32mINFO←[0m:     Started server process [←[36m29340←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
```

Run frontend:
```
cd pizza-violation-frontend
npm install  # this used to install the package dependencies
npm run start
```
-this will show up

```
  VITE v7.3.0  ready in 1149 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```
-open http://localhost:5173/ and that is the project

Notes:
- Run both backend and frontend in separate terminals.
- make sure the backend start first then the front as i did in the video
- If `start` (Windows) doesn't apply on your OS, run `uvicorn fast_api.api:app --reload --port 8000` directly.

---

## extra informations
- i used go2rtc to locally stream an rtsp instead of real camera and its easy to download and easy to launch
- go2rtc zip has 2 files the exe and the yaml the yaml will be including the type of rtsp you want the stream to be and the location of the video you want to stream
- make sure to change the directories to your the video you want to stream 
