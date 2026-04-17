# Rare Disease X-Ray Final Project

## Project Title
Domain-Adaptive Synthetic Data Generation for Rare Disease Chest X-Ray Classification Using Conditional Diffusion Models

## Project Structure

```
./
│
├── frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── home.py
│   │   ├── preprocessing.py
│   │   ├── generate.py
│   │   ├── classify.py
│   │   └── explainability.py
│   └── components/
│       ├── sidebar.py
│       └── preview_card.py
│
├── backend/
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── preprocessing_service.py
│   │   ├── diffusion_service.py
│   │   ├── classification_service.py
│   │   └── gradcam_service.py
│   └── schemas/
│       └── api_models.py
│
├── ml/
│   ├── preprocessing/
│   │   └── data_preprocessing.py
│   ├── diffusion/
│   │   ├── conditional_unet.py
│   │   ├── dataset_loader.py
│   │   ├── train_ddpm.py
│   │   └── generate_samples.py
│   ├── classification/
│   │   ├── efficientnet_model.py
│   │   ├── train_classifier.py
│   │   └── evaluate.py
│   └── explainability/
│       └── gradcam.py
│
├── outputs/
│   ├── checkpoints/
│   ├── previews/
│   ├── synthetic_data/
│   ├── classifier_results/
│   └── gradcam/
│
├── configs/
│   └── config.py
│
├── utils/
│   └── helper.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Open the project folder in VS Code.
2. Install dependencies using Python:

```bash
python -m pip install -r requirements.txt
```

If package installation fails for torch on Python 3.13, install the compatible wheel explicitly:

```bash
python -m pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

Alternatively use the helper batch script on Windows:

```bash
install_requirements.bat
```

## Run the App

Start the Streamlit UI:

```bash
python -m streamlit run frontend/app.py
```

Then use the sidebar pages as a simple website flow:

1. Home - overview and pipeline flow.
2. Preprocessing - prepare your dataset.
3. Generate - choose a disease name and synthesize X-rays.
4. Classify - upload an X-ray and see the predicted disease name.
5. Explainability - inspect Grad-CAM attention for a chosen disease.

Start the backend API:

```bash
python -m uvicorn backend.api.routes:app --reload
```

Run the full pipeline:

```bash
python main.py --mode full_pipeline
```

Windows helper scripts:

```bash
run_frontend.bat
run_backend.bat
run_full_pipeline.bat
```

## Notes

- `frontend/` contains the Streamlit demo pages.
- `backend/` contains FastAPI endpoints for preprocessing, diffusion, classification, Grad-CAM, and metrics.
- `ml/` contains preprocessing, diffusion, classification, and explainability modules.

> Note: If you use a custom dataset with different disease classes, update `CLASS_NAMES` in `configs/config.py` so the Generate, Classify, and Explainability pages show names instead of indices.
- `outputs/` stores model checkpoints, synthetic images, classifier results, Grad-CAM outputs, and preview artifacts.

## Expected Flow

1. Dataset Collection + Preprocessing
2. Conditional DDPM Diffusion Training
3. Synthetic Chest X-ray Generation
4. EfficientNet Classification
5. Grad-CAM Visualization
6. F1-score Comparison
