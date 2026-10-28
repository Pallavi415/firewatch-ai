import sys
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Configure paths to use relative directory structures for deployment portability
PROJECT_DIR = Path(".")
MODEL_PATH = PROJECT_DIR / "models/best.pt"
OUTPUT_DIR = PROJECT_DIR / "outputs/dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_DIR / "src"))
from inference import analyze_image, analyze_video

st.set_page_config(
    page_title="FireWatch AI | Safety Command Center",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#080d18;color:#eef2ff}
[data-testid="stSidebar"] {background:#0d1422}
.block-container {padding-top:1.5rem;max-width:1500px}
.hero {padding:22px;border:1px solid #293951;border-radius:18px;
background:linear-gradient(110deg,#19263b,#101827);margin-bottom:20px}
.card {background:#121d2e;border:1px solid #293951;border-radius:14px;
padding:18px;margin-bottom:10px}
.small {color:#94a3b8;font-size:12px}
.big {font-size:28px;font-weight:800;margin-top:8px}
div.stButton>button {background:#f97345;color:white;border:0;
border-radius:9px;font-weight:700;min-height:42px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔥 FireWatch AI")
    st.caption("SAFETY INTELLIGENCE")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Video Analysis", "Risk Reports", "Model Information"]
    )
    st.divider()
    st.success("Dashboard Online")
    st.caption("YOLO · Fire / Smoke")
    st.markdown("⚡ **Colab Sandbox Helper**")
    st.info("If your browser blocks download buttons inside the iframe, run this code block in a Colab notebook cell to retrieve files:\n\n`from google.colab import files`\n`files.download('/content/drive/MyDrive/FireWatch-AI/outputs/dashboard/annotated_video.mp4')`")

st.markdown("""
<div class="hero">
  <div style="color:#ff9b66;letter-spacing:2px;font-size:11px">
  INDUSTRIAL SAFETY INTELLIGENCE</div>
  <h1>FireWatch Command Center</h1>
  <p>Analyze footage. Detect hazards. Review model outputs.</p>
</div>
""", unsafe_allow_html=True)

if page in ["Overview", "Video Analysis"]:
    st.subheader("Video & Image Analysis")
    st.caption("Upload prerecorded footage and run the existing trained model.")

    input_type = st.radio("Input type", ["Video", "Image"], horizontal=True)
    file_types = ["mp4", "avi", "mov", "mkv"] if input_type == "Video" else [
        "jpg", "jpeg", "png", "webp"
    ]
    uploaded = st.file_uploader("Choose a file", type=file_types)

    conf = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)

    if uploaded:
        st.markdown("#### Input preview")
        if input_type == "Video":
            st.video(uploaded)
        else:
            st.image(uploaded, use_container_width=True)

        if st.button("⚡ Analyze with FireWatch AI", type="primary"):
            suffix = Path(uploaded.name).suffix.lower()
            input_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    input_path = Path(tmp.name)

                with st.spinner("Loading model and analyzing input... ⏳"):
                    if not MODEL_PATH.is_file():
                        st.error(f"Trained model not found: {MODEL_PATH}")
                        st.stop()

                    model = YOLO(str(MODEL_PATH))

                    if input_type == "Image":
                        output_path = OUTPUT_DIR / "annotated_image.jpg"
                        result = analyze_image(
                            model, input_path, output_path, conf
                        )
                        rows = result["detections"]
                        fire_count = result["fire_count"]
                        smoke_count = result["smoke_count"]
                        frames = 1
                    else:
                        output_path = OUTPUT_DIR / "annotated_video.mp4"
                        result = analyze_video(
                            model, input_path, output_path, conf
                        )
                        rows = result["detections"]
                        fire_count = sum(r["class"] == "fire" for r in rows)
                        smoke_count = sum(r["class"] == "smoke" for r in rows)
                        frames = result["frames_processed"]

                    csv_path = OUTPUT_DIR / "detections.csv"
                    pd.DataFrame(rows).to_csv(csv_path, index=False)

                    # Save state immediately
                    st.session_state["fw_result"] = {
                        "type": input_type,
                        "name": output_path.name,
                        "rows": rows,
                        "fire_count": fire_count,
                        "smoke_count": smoke_count,
                        "frames": frames,
                    }

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                if input_path and input_path.exists():
                    input_path.unlink()

    result = st.session_state.get("fw_result")
    if result:
        st.divider()
        st.subheader("Analysis Results")

        a, b, c = st.columns(3)
        a.metric("Fire detections", result["fire_count"])
        b.metric("Smoke detections", result["smoke_count"])
        c.metric("Frames processed", result["frames"])

        output_path = OUTPUT_DIR / result["name"]
        if output_path.exists():
            st.markdown("#### Annotated output")
            if result["type"] == "Image":
                st.image(str(output_path), use_container_width=True)
                mime = "image/jpeg"
            else:
                st.video(str(output_path))
                mime = "video/mp4"

            st.download_button(
                "⬇ Download annotated output",
                data=output_path.read_bytes(),
                file_name=result["name"],
                mime=mime,
            )

        rows = result["rows"]
        if rows:
            st.markdown("#### Detection details")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No detections found at the selected confidence threshold.")

        csv_path = OUTPUT_DIR / "detections.csv"
        if csv_path.exists():
            st.download_button(
                "⬇ Download detection CSV",
                data=csv_path.read_bytes(),
                file_name="firewatch_detections.csv",
                mime="text/csv",
            )

elif page == "Risk Reports":
    st.subheader("Risk Reports")
    st.info("Awaiting risk evaluation setup.")

elif page == "Model Information":
    st.subheader("Model Information")
    st.write("Architecture: YOLOv8")
    st.write("Classes: fire, smoke")
    st.write("Weights:", str(MODEL_PATH))
    st.write("Model available:", MODEL_PATH.is_file())
    st.caption("This dashboard provides decision support, not certified emergency detection.")
