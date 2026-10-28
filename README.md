# FireWatch AI 🔥

FireWatch AI is an educational computer-vision prototype for reviewing prerecorded fire and smoke footage. It uses a custom-trained YOLO26n model and a Streamlit interface to analyze uploaded images and videos, display bounding-box annotations, and export detection records.

> **Project status:** Prototype. Image/video inference is implemented. Risk Reports is currently a placeholder in the Streamlit app. PPE detection, firefighter proximity analysis, and validated fire-progression assessment are not integrated into the app. This project is not a certified alarm or life-safety system.

## Features

- Upload images: JPG, JPEG, PNG, WEBP.
- Upload prerecorded videos: MP4, AVI, MOV, MKV.
- Run the included fire/smoke detector with an adjustable confidence threshold.
- View annotated output with bounding boxes.
- Review predicted classes, confidence scores, bounding-box coordinates, and video frame numbers.
- Download annotated media and detection CSV files.
- Use the reusable inference functions in `src/inference.py`.

## Current architecture

```text
firewatch-ai/
├── app/
│   └── streamlit_app.py       # Streamlit interface
├── src/
│   └── inference.py           # Image and video inference
├── models/
│   └── best.pt                # Trained fire/smoke model weights
├── runs/                      # Training and evaluation artifacts
├── reports/                   # Saved model metrics and comparisons
├── outputs/                   # Example outputs and evaluation artifacts
├── configs/                   # Experiment configuration
├── requirements.txt           # Python dependencies
├── packages.txt               # Linux system packages for hosting
└── .gitignore
```

## Model

The included `models/best.pt` checkpoint is a custom-trained **YOLO26n** fire/smoke detector. The saved model-loading output identifies the classes as `fire` and `smoke`.

The repository contains a saved metrics report at `reports/fire_smoke_test_metrics.json`. It reports precision 90.71%, recall 82.73%, mAP@50 87.07%, and mAP@50–95 59.71%. These are reported experiment results and have **not been independently reproduced as part of this README update**. See the evaluation artifacts and dataset/split limitations before drawing conclusions from them.

Detection counts are counts of model predictions. For video, the same real-world fire or smoke region may be detected in many frames, so prediction counts are not unique incident counts.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Pallavi415/firewatch-ai.git
cd firewatch-ai
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Use a Python version supported by the installed PyTorch and Ultralytics versions, then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

FFmpeg is used to re-encode processed video when available. `packages.txt` lists system packages intended for Linux hosting environments. Local installation steps vary by operating system.

### 4. Start the dashboard

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app expects model weights at `models/best.pt`, relative to the repository root. Keep that file in place when running the app. Output files are written under `outputs/dashboard/`.

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub account that can access this repository.
2. Select **Create app** and choose the `Pallavi415/firewatch-ai` repository, the `main` branch, and `app/streamlit_app.py` as the main file.
3. Deploy the app. Streamlit Community Cloud installs Python dependencies from `requirements.txt` and can use `packages.txt` for supported Linux packages.
4. Confirm that the deployed environment can install the dependencies and load `models/best.pt`. Check the app logs if startup or model loading fails.
5. Test both an image and a short video after deployment.

The repository can remain private if your Streamlit Community Cloud account and deployment setup have permission to access it. Do not make the repository public just to attempt deployment. If the app fails because of model compatibility, system packages, memory, or dependency issues, resolve that from the deployment logs before sharing the app link.

## Current limitations

- **Risk Reports:** Placeholder page in the Streamlit application; the experimental notebook-based risk calculations are not integrated into the app.
- **PPE and proximity:** PPE compliance detection and firefighter-to-hazard proximity estimation are not integrated into the current app.
- **Temporal progression:** Video is processed frame by frame, but the current inference module does not confirm incidents across time or reliably determine physical fire growth.
- **Evaluation:** Saved metrics and small video experiments are not a substitute for a reproducible, representative held-out evaluation.
- **Video counts:** Fire/smoke counts in the app count prediction rows across frames, not unique objects or incidents.
- **Video encoding:** FFmpeg re-encoding is attempted; if it fails, the fallback output may not be browser-compatible.
- **Deployment:** The app needs the trained model file and compatible dependencies. Deployment success must be confirmed by running the app in the target environment.
- **Safety:** The model can produce false positives and miss hazards. Do not use it as a replacement for certified alarms, trained personnel, emergency procedures, or professional safety systems.

## Development roadmap

- Make evaluation reproducible with a documented dataset split, checkpoint, and evaluation command.
- Add automated tests for inference and video-output handling.
- Integrate a tested risk-analysis module into the Streamlit interface.
- Evaluate temporal confirmation, fire progression, PPE detection, and proximity estimation on representative data before describing them as supported features.
- Add CI checks and document the model/data provenance and applicable licenses.

## Data, weights, and responsible use

The `datasets/` directory is excluded by `.gitignore`; datasets are not required to launch the dashboard, but the included checkpoint is required for inference. Before redistributing this project, verify the licenses and redistribution terms for the training data, sample media, pretrained components, dependencies, and trained weights. Do not commit credentials, access tokens, private datasets, or local environment files.

## Disclaimer

FireWatch AI is an educational development prototype for decision support. It is not certified for fire alarms, emergency response, or life-safety decisions.

## Author

**Pallavi Dahiya** · [GitHub: @Pallavi415](https://github.com/Pallavi415)
