from insightface.app import FaceAnalysis
import cv2
import numpy as np
import time

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

MATCH_THRESHOLD = 0.62   # Best accuracy range = 0.58 – 0.70
FRAME_SAMPLES = 5        # Number of frames to verify same result

# ------------------ LOAD REFERENCE FACE ------------------
ref_img = cv2.imread("reference.jpg")
if ref_img is None:
    print("❌ reference.jpg missing in folder!")
    exit()

ref_face = app.get(ref_img)
if len(ref_face) == 0:
    print("❌ No face found in reference image!")
    exit()

ref_emb = ref_face[0].normed_embedding
print("\n📌 Reference Loaded • Threshold =", MATCH_THRESHOLD)

# ------------------ LIVE CAMERA ------------------
cap = cv2.VideoCapture(0)
print("\n🎥 Live Verification Started\nPress 'V' to verify | Press 'Q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read fail")
        break

    faces = app.get(frame)
    display = frame.copy()

    if len(faces) > 0:
        live_emb = faces[0].normed_embedding
        sim = np.dot(ref_emb, live_emb)

        # On-screen display
        color = (0,255,0) if sim > MATCH_THRESHOLD else (0,0,255)
        cv2.putText(display, f"Similarity: {sim:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Verification Window", display)
    key = cv2.waitKey(1) & 0xFF

    # --------- VERIFY ON KEY PRESS (multi-frame logic) ---------
    if key == ord('v'):
        scores = []
        print("\n🔍 Capturing multiple frames... stay still 2 sec")

        for _ in range(FRAME_SAMPLES):
            ret, frame = cap.read()
            faces = app.get(frame)
            if len(faces) > 0:
                scores.append(np.dot(ref_emb, faces[0].normed_embedding))
            time.sleep(0.15)  # slight delay to stabilize face

        if len(scores) == 0:
            print("❌ No face detected — try again")
            continue

        avg_score = np.mean(scores)
        print(f"Avg Similarity = {avg_score:.3f}")

        if avg_score > MATCH_THRESHOLD:
            print("✔ Verified — Same Person\n")
        else:
            print("❌ Verification Failed — Face Not Matching\n")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
