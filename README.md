Sign Language Recognition API
This project is an AI-powered API designed to recognize sign language and translate it into text in real-time. It was developed as part of a graduation project at the Egyptian E-Learning University (EELU).
Overview:
The system utilizes deep learning models to process hand and body landmarks, providing accurate sign-to-text translations. The API is built with FastAPI and deployed on Railway for easy integration with web and mobile applications.
Tech Stack:
Framework: FastAPI

Deep Learning: PyTorch

Computer Vision: MediaPipe (Holistic landmarks)

Deployment: Railway

Dataset: WLASL (World-Level American Sign Language)

Project Structure:
app.py: Main server file and API endpoints.

best_v6.pt: The trained PyTorch model weights.

label_map_v6.json: Mapping file for sign labels.

requirements.txt: Python dependencies.

Procfile: Configuration for production deployment.
