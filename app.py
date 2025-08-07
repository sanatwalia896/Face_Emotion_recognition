# Refactor the user's app.py to use streamlit-webrtc instead of cv2.VideoCapture for webcam access in Streamlit Cloud

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import streamlit as st
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Load model
class Deep_Emotion(nn.Module):
    def __init__(self):
        super(Deep_Emotion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Deep_Emotion()
model.load_state_dict(torch.load('Face_Emotion_Recognition_2.pth', map_location=torch.device('cpu')))
model.eval()

emotions = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# UI functions (happiness, sadness, etc.)
def show_emotion_quiz(emotion, key):
    if emotion == "Happy":
        st.info("Detected: Happy 😊")
        show_happiness_features(key)
    elif emotion == "Sad":
        st.info("Detected: Sad 😔")
        show_sadness_support(key)
    elif emotion == "Neutral":
        st.info("Detected: Neutral 😐")
        show_neutral_quiz(key)
    elif emotion == "Angry":
        st.info("Detected: Angry 😠")
        show_anger_support(key)
    elif emotion == "Disgust":
        st.info("Detected: Disgust 😖")
        show_disgust_support(key)
    elif emotion == "Fear":
        st.info("Detected: Fear 😨")
        show_fear_support(key)
    elif emotion == "Surprise":
        st.info("Detected: Surprise 😲")
        show_surprise_support(key)

# Emotion predictor
def predict_emotion(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted.item()]

# Video processor class for webrtc
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_advice_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5)
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            if roi.size != 0:
                emotion = predict_emotion(roi)
                cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                if time.time() - self.last_advice_time > 15:
                    self.last_advice_time = time.time()
                    st.session_state['last_emotion'] = (emotion, self.last_advice_time)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Replace OpenCV capture with webrtc_streamer
def capture_webcam():
    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# Embed YouTube video
def stream_video(youtube_link):
    st.subheader("YouTube Video")
    st.write(f"YouTube Video Link: {youtube_link}")
    st.markdown(f'<iframe width="640" height="360" src="https://www.youtube.com/embed/{youtube_link.split("=")[-1]}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

# Main app
def main():
    st.title("Face Emotion Recognition App with WebRTC")
    youtube_link = st.sidebar.text_input("Enter YouTube video link:")

    if youtube_link:
        stream_video(youtube_link)
        st.subheader("Webcam Feed")
        capture_webcam()

        if 'last_emotion' in st.session_state:
            emotion, key = st.session_state['last_emotion']
            show_emotion_quiz(emotion, str(key))
    else:
        st.warning("Please enter a YouTube video link.")

main()
