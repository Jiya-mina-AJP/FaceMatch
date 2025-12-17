import atexit
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string
from insightface.app import FaceAnalysis

# -------------------- Flask --------------------
app = Flask(__name__)

# -------------------- Config --------------------
MATCH_THRESHOLD = 0.62
MODEL_NAME = "buffalo_l"

# -------------------- Load Model --------------------
print("🔄 Loading InsightFace model...")
face_app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Model loaded successfully")

# -------------------- Shared State --------------------
ref_embedding = None
ref_lock = threading.Lock()
camera = None
camera_lock = threading.Lock()


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
        return camera


def compute_embedding(img):
    faces = face_app.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding


# -------------------- UI --------------------
HTML = """
<!doctype html>
<html>
<head>
<title>Face Verification</title>
<style>
body { font-family: Arial; background:#f4f6fb; padding:30px; }
.card { background:white; padding:20px; border-radius:10px; max-width:900px; margin:auto; }
#status { font-weight:bold; margin-top:10px; }
</style>
</head>
<body>
<div class="card">
<h2>Live Face Verification</h2>

<form id="uploadForm">
<input type="file" name="reference" required>
<button type="submit">Upload Reference</button>
</form>

<p id="status">Waiting for reference image...</p>

<img src="/video_feed" width="720">
</div>

<script>
document.getElementById("uploadForm").onsubmit = async function(e){
    e.preventDefault();
    let formData = new FormData(this);
    document.getElementById("status").innerText = "Uploading reference...";
    let res = await fetch("/set_reference", {method:"POST", body:formData});
    let text = await res.text();
    document.getElementById("status").innerText = text;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/set_reference", methods=["POST"])
def set_reference():
    global ref_embedding

    file = request.files.get("reference")
    if not file:
        return "❌ No file uploaded"

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "❌ Invalid image"

    emb = compute_embedding(img)
    if emb is None:
        return "❌ No face detected in reference image"

    with ref_lock:
        ref_embedding = emb

    return "✅ Reference uploaded. Live scanning started!"


def gen_frames():
    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        display = frame.copy()

        with ref_lock:
            ref_emb = ref_embedding

        if ref_emb is None:
            cv2.putText(display, "Upload reference image",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        else:
            faces = face_app.get(frame)
            if faces:
                sim = float(np.dot(ref_emb, faces[0].normed_embedding))
                matched = sim > MATCH_THRESHOLD
                color = (0, 255, 0) if matched else (0, 0, 255)
                text = "✔ MATCH" if matched else "✖ NO MATCH"
                cv2.putText(display, f"{text}  ({sim:.2f})",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                cv2.putText(display, "No face detected",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", display)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@atexit.register
def cleanup():
    global camera
    if camera and camera.isOpened():
        camera.release()


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        use_reloader=False
    )

