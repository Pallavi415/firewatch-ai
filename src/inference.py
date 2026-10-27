
import subprocess
from pathlib import Path
import cv2
from ultralytics import YOLO


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def analyze_image(model, image_path, output_path, conf=0.25):
    image_path = str(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save annotated image: {output_path}")

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append({
            "class": model.names[class_id],
            "confidence": round(float(box.conf[0].item()), 4),
            "x1": round(float(box.xyxy[0][0].item()), 2),
            "y1": round(float(box.xyxy[0][1].item()), 2),
            "x2": round(float(box.xyxy[0][2].item()), 2),
            "y2": round(float(box.xyxy[0][3].item()), 2),
        })

    return {
        "output_path": str(output_path),
        "detections": detections,
        "fire_count": sum(d["class"] == "fire" for d in detections),
        "smoke_count": sum(d["class"] == "smoke" for d in detections),
    }


def analyze_video(model, video_path, output_path, conf=0.25):
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    # Save raw OpenCV output to a temporary path first
    temp_output_path = output_path.with_name("temp_" + output_path.name)

    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create temporary output video: {temp_output_path}")

    frame_count = 0
    fire_frames = 0
    smoke_frames = 0
    detection_rows = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = result.plot()
            writer.write(annotated)
            frame_count += 1

            classes_in_frame = set()
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = model.names[class_id]
                classes_in_frame.add(label)
                detection_rows.append({
                    "frame": frame_count,
                    "class": label,
                    "confidence": round(float(box.conf[0].item()), 4),
                    "x1": round(float(box.xyxy[0][0].item()), 2),
                    "y1": round(float(box.xyxy[0][1].item()), 2),
                    "x2": round(float(box.xyxy[0][2].item()), 2),
                    "y2": round(float(box.xyxy[0][3].item()), 2),
                })

            fire_frames += int("fire" in classes_in_frame)
            smoke_frames += int("smoke" in classes_in_frame)
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise RuntimeError("No readable frames found in uploaded video.")

    # Use ffmpeg to re-encode the temp video to H.264 (AVC) so it plays on web/Streamlit
    try:
        if output_path.exists():
            output_path.unlink()
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_output_path),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            str(output_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        # Fallback to the unencoded video if ffmpeg fails
        if temp_output_path.exists():
            temp_output_path.rename(output_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    return {
        "output_path": str(output_path),
        "frames_processed": frame_count,
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "detections": detection_rows,
    }
