# FireWatch AI 🔥

**FireWatch AI** is a computer-vision project for reviewing prerecorded fire-response and industrial-safety footage. It provides a Streamlit dashboard that runs a trained YOLO model on uploaded images or videos, visualizes detections, and lets users download annotated media and detection records.

> **Status:** The repository currently contains the Streamlit interface and reusable image/video inference code. The trained weights are expected at a local Google Drive path and are not included in this repository. Risk reporting, temporal risk assessment, PPE analysis, and a complete deployment workflow should be considered unfinished unless implemented and tested separately.

## Features

- **Image analysis:** Upload an image and run object detection with a configurable confidence threshold.
- **Video analysis:** Upload prerecorded footage and process its frames.
- **Visual results:** View annotated images/videos with model bounding boxes.
- **Detection summaries:** Review fire and smoke detection counts and frame totals.
- **Detection export:** Download detection details as a CSV file.
- **Streamlit interface:** Navigate Overview, Video Analysis, Risk Reports, and Model Information pages.
- **Reusable inference functions:** Image and video processing is separated into `src/inference.py`.

## Current implementation

| Component | Responsibility |
|---|---|
| `app/streamlit_app.py` | Streamlit user interface, uploads, model invocation, results, and downloads |
| `src/inference.py` | YOLO inference for images/videos, bounding-box extraction, annotation, and video output |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages used by the hosting environment |
| `.gitignore` | Excludes Python cache files, environment files, and selected generated data |

### Detection output

The inference module records the predicted class, confidence, and bounding-box coordinates (`x1`, `y1`, `x2`, `y2`). Video records also include the frame number. The dashboard saves a CSV named `detections.csv` and an annotated image or video under its configured output directory.

Detection counts are counts of predictions, not necessarily counts of unique real-world fire or smoke events. Model performance depends on the trained weights, data, confidence threshold, and recording conditions.

## Technology stack

- Python
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV
- Streamlit
- Pandas
- FFmpeg for video re-encoding when available

## Repository structure

```text
firewatch-ai/
├── app/
│   └── streamlit_app.py
├── src/
│   └── inference.py
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pallavi415/firewatch-ai.git
cd firewatch-ai
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

FFmpeg and the operating-system packages listed in `packages.txt` may also be needed, particularly for video encoding and hosted deployments. Installation commands vary by operating system and hosting platform.

## Model configuration

The current dashboard expects trained weights at:

```text
/content/drive/MyDrive/FireWatch-AI/runs/fire_smoke/yolo26n_fire_smoke_v1/weights/best.pt
```

This is a Google Colab/Google Drive-specific path. On another machine or hosting service, update `PROJECT_DIR` and `MODEL_PATH` in `app/streamlit_app.py` to locations that exist in that environment, and ensure that the trained weights are available there.

The current inference code expects a YOLO-compatible model with class names that include `fire` and `smoke` for the corresponding counts. Actual class names are determined by the supplied weights. The dashboard does not include the model weights or training dataset in this repository.

## Run the dashboard

After installing the dependencies and configuring a valid model path, run:

```bash
streamlit run app/streamlit_app.py
```

Open the local URL shown by Streamlit. Upload a supported image or video, choose the confidence threshold, and select **Analyze with FireWatch AI**. After processing, view the annotated output and download the media or CSV results.

### Supported inputs in the current UI

- **Images:** JPG, JPEG, PNG, WEBP
- **Videos:** MP4, AVI, MOV, MKV

Video processing time depends on the video's length/resolution and the available compute.

## Limitations and safety

- This is a prototype for experimentation and decision support, **not a certified fire-alarm, emergency-response, or life-safety system**.
- Results depend on the quality and scope of the trained model and may include false positives or missed detections.
- The dashboard's Risk Reports page is currently a placeholder; a validated risk-scoring engine is not implemented in the inspected application code.
- PPE compliance detection and firefighter proximity assessment are not implemented in the inspected inference module.
- Although temporal video processing is performed frame by frame, event confirmation, progression tracking, and temporal smoothing are not implemented in the inspected inference module.
- Validate the model on representative, held-out data before relying on its predictions.
- Avoid committing private datasets, large video files, trained weights, credentials, or tokens.

## Roadmap

Potential next steps for the project include:

- Add a tested risk-scoring pipeline and explainable risk reports.
- Add temporal confirmation and fire progression analysis.
- Integrate and evaluate PPE detection and firefighter-to-hazard proximity estimation.
- Add reproducible evaluation reports with clearly defined metrics.
- Make model and output paths configurable through environment variables or a settings file.
- Add automated tests and a documented deployment workflow.

These items are proposed future work, not claims of existing functionality.

## Disclaimer

FireWatch AI is an educational and development project. It must not replace trained personnel, established safety procedures, certified alarms, or professional emergency systems.

## Author

**Pallavi Dahiya**  
GitHub: [@Pallavi415](https://github.com/Pallavi415)

---

If you use or extend this project, document the model source, training data, dataset licenses, and evaluation results alongside the relevant code.
