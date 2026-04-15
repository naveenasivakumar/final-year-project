# Medical Diffusion App

## Project Title

**Domain-Adaptive Synthetic Data Generation for Rare Disease Chest X-Ray Classification Using Conditional Diffusion Models**

## Project Structure

```text
medical_diffusion_app/
+-- frontend/
|   +-- app.py
|   +-- pages/
|   +-- components/
+-- backend/
|   +-- api/
|   +-- services/
|   +-- schemas/
+-- ml/
|   +-- models/
|   +-- training/
|   +-- generation/
|   +-- data/
+-- utils/
+-- configs/
+-- outputs/
+-- main.py
+-- requirements.txt
+-- requirements-ml.txt
+-- README.md
```

## What This Project Does

- `frontend/` contains a simple Streamlit medical UI.
- `backend/` exposes FastAPI routes for health, generation, and training.
- `ml/` keeps the diffusion architecture, training logic, data loader, and generation flow reusable.
- `outputs/` stores checkpoints, preview grids, synthetic images, and logs in one place.

## Dataset Layout

Create a dataset folder inside `medical_diffusion_app/dataset/` like this:

```text
dataset/
+-- normal/
+-- pleural_effusion/
+-- cardiomegaly/
+-- pneumonia/
+-- atelectasis/
+-- consolidation/
```

## Installation

Run these commands from the repo root:

```bash
.venv\Scripts\python.exe -m pip install -r medical_diffusion_app/requirements.txt
.venv\Scripts\python.exe -m pip install -r medical_diffusion_app/requirements-ml.txt
```

If you only want the frontend and backend shell to start without ML generation, the first command is enough.

## Run Commands

Backend:

```bash
.venv\Scripts\python.exe -m uvicorn medical_diffusion_app.backend.api.routes:app --reload
```

Frontend:

```bash
.venv\Scripts\python.exe -m streamlit run medical_diffusion_app/frontend/app.py
```

Training:

```bash
.venv\Scripts\python.exe -m medical_diffusion_app.main --mode train
```

Generation:

```bash
.venv\Scripts\python.exe -m medical_diffusion_app.main --mode generate --label 1 --num_samples 4
```

## Notes

- The project now uses package-style imports, so commands work from the repo root.
- If ML dependencies are missing, the app shows a clear install message instead of a Python traceback.
- Once training creates `medical_diffusion_app/outputs/checkpoints/conditional_ddpm_best.pt`, the backend automatically uses it for generation.
