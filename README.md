# FaceMatch: Real-Time Facial Verification System

A high-performance, 1:1 facial verification application designed for secure access control. This system leverages Deep Learning to map facial features into a numerical space, allowing for precise identity matching against a reference database.

## Project Overview
Identity theft and unauthorized access are critical security concerns. This project provides a real-time solution by comparing live camera feeds against a "Ground Truth" reference image. Unlike simple face detection, this system performs **Feature Extraction** to identify individuals even under varying lighting conditions or slight changes in appearance (e.g., glasses or facial hair).

## Key Technical Insights
Through model calibration and real-world testing, the system's reliability was optimized using:
1. **512-D Embeddings:** Captured deep facial features that remain invariant to minor physical changes.
2. **0.62 Similarity Threshold:** Identified as the "Sweet Spot" to minimize False Acceptances while maintaining high user convenience.
3. **Temporal Averaging:** Aggregating 5 frames significantly reduced "noise" and motion blur errors common in single-shot verification.

## Tech Stack
* **Language:** Python
* **AI Engine:** InsightFace (ArcFace Architecture), buffalo_l model
* **Inference:** ONNX Runtime (Optimized for CPU)
* **Mathematics:** NumPy (Vectorized Dot Products)
* **Computer Vision:** OpenCV
* **Web Deployment:** Flask, HTML5, Vanilla JavaScript (Fetch API)

## Machine Learning Architecture
* **Detection & Alignment:** Utilized InsightFace to locate faces and normalize their position within a 640x640 coordinate space.
* **Feature Extraction:** Generated normalized 512-dimensional embeddings, transforming visual data into a mathematical "fingerprint."
* **Matching Logic:** Implemented **Cosine Similarity** via Dot Product calculations. A match is triggered only if the similarity score exceeds the calibrated 0.62 threshold.
* **Stability Engineering:** Developed a multi-frame validation loop with a 150ms delay between samples to ensure data diversity and high-confidence verification.
* **Concurrency:** Used **Threading Locks** in the Flask backend to manage shared resources (camera and reference image) across multiple web requests.

## Repository Structure
```text
├── templates/
│   └── index.html          # Web interface for reference upload and live monitoring
├── web_app.py              # Flask server handling MJPEG streaming and AI inference
├── reference.jpg           # Ground Truth image used for identity comparison
├── requirements.txt        # Project dependencies (InsightFace, ONNX, etc.)
└── model_training_notes.md # Documentation on threshold calibration and model selection
