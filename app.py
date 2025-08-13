# app.py — Streamlit WebRTC Emotion Dashboard (polished UI + live charts)

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

# -----------------------------
# Page config & basic styling
# -----------------------------
st.set_page_config(page_title="🎭 Emotion Recognition Dashboard", layout="wide", page_icon="🎭")
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    h1, h2, h3 { font-family: 'Trebuchet MS', system-ui, -apple-system, sans-serif; }
    .card {
        border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    .muted { color: #6b7280; }
    .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 0.85rem; background:#f3f4f6;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Model
# -----------------------------
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
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = x.view(-1, 32 * 10 * 10)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = self.fc3(x)
        return x

@st.cache_resource(show_spinner=False)
def load_model():
    m = Deep_Emotion()
    # Put your checkpoint path here (kept same name as yours)
    m.load_state_dict(torch.load('Face_Emotion_Recognition_2.pth', map_location=torch.device('cpu')))
    m.eval()
    return m

model = load_model()

emotions = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

emotion_colors = {
    "Happy": "#FFD54F",      # warm yellow
    "Sad": "#64B5F6",        # calm blue
    "Neutral": "#B0BEC5",    # soft gray
    "Angry": "#EF5350",      # red
    "Disgust": "#8BC34A",    # green
    "Fear": "#AB47BC",       # purple
    "Surprise": "#4DD0E1"    # teal
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# UI helpers
# -----------------------------
def emotion_pill(emotion: str):
    color = emotion_colors.get(emotion, "#f3f4f6")
    st.markdown(
        f"<span class='pill' style='background:{color}; color:#111827'>{emotion}</span>",
        unsafe_allow_html=True
    )

def card(title: str, body: str, bg: str = "#ffffff"):
    st.markdown(
        f"""
        <div class="card" style="background:{bg}">
            <h4 style="margin:0 0 6px 0">{title}</h4>
            <div class="muted">{body}</div>
        </div>
        """, unsafe_allow_html=True
    )

# Emotion-specific quizzes (kept simple & styled)
def show_happiness_features(key):
    card("Happiness Tip", "Smiling is contagious. Keep it up!", "#FFF8E1")
    ans = st.radio("Which is a feature of happiness?", ["Smiling", "Crying", "Frowning", "None"], key=f"{key}_happy", index=None)
    if ans == "Smiling": st.success("Correct! 😄")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_sadness_support(key):
    card("It's okay to feel sad", "Reach out to someone you trust. A short walk helps too.", "#E3F2FD")
    ans = st.radio("Synonym for 'sad'?", ["Happy", "Gloomy", "Excited", "None"], key=f"{key}_sad", index=None)
    if ans == "Gloomy": st.success("Correct! 🌧️")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_neutral_quiz(key):
    card("Neutral", "Staying even-keeled can be powerful.", "#ECEFF1")
    ans = st.radio("Synonym for 'neutral'?", ["Indifferent", "None"], key=f"{key}_neutral", index=None)
    if ans == "Indifferent": st.success("Correct! 😐")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_anger_support(key):
    card("Anger Support", "Try a 4–7–8 breathing cycle and count backwards from 20.", "#FFEBEE")
    ans = st.radio("Good way to manage anger?", ["Punching a wall", "Taking deep breaths", "Yelling", "None"], key=f"{key}_anger", index=None)
    if ans == "Taking deep breaths": st.success("Correct! 🧘")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_disgust_support(key):
    card("Disgust", "Noticing and naming the feeling helps you move through it.", "#F1F8E9")
    ans = st.radio("What might elicit disgust?", ["Fav food", "Rotten item", "Funny movie", "None"], key=f"{key}_disgust", index=None)
    if ans == "Rotten item": st.success("Correct! 😖")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_fear_support(key):
    card("Fear Support", "Ground yourself: name 5 things you can see, 4 you can feel, etc.", "#F3E5F5")
    ans = st.radio("What helps overcome fear?", ["Avoiding it", "Face gradually", "Ignore it", "None"], key=f"{key}_fear", index=None)
    if ans == "Face gradually": st.success("Correct! 🧗")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_surprise_support(key):
    card("Surprise", "Take a moment, breathe, then respond.", "#E0F7FA")
    ans = st.radio("Common reaction to surprise?", ["Gasping", "None"], key=f"{key}_surprise", index=None)
    if ans == "Gasping": st.success("Correct! 😲")
    elif ans == "None": st.info("No selection made.")
    elif ans: st.error("Incorrect. Try again.")

def show_emotion_quiz(emotion, key):
    st.caption("Last detected emotion:")
    emotion_pill(emotion)
    st.markdown("---")
    if emotion == "Happy":
        show_happiness_features(key)
    elif emotion == "Sad":
        show_sadness_support(key)
    elif emotion == "Neutral":
        show_neutral_quiz(key)
    elif emotion == "Angry":
        show_anger_support(key)
    elif emotion == "Disgust":
        show_disgust_support(key)
    elif emotion == "Fear":
        show_fear_support(key)
    elif emotion == "Surprise":
        show_surprise_support(key)

# -----------------------------
# Inference helpers
# -----------------------------
def predict_emotion(face_img_bgr: np.ndarray) -> str:
    # Resize before transform to avoid aspect issues on tiny crops
    face_img_bgr = cv2.resize(face_img_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    face_tensor = transform(face_img_bgr).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted.item()]

# -----------------------------
# WebRTC Video Processor
# -----------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_advice_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            if roi.size != 0:
                emotion = predict_emotion(roi)
                # Draw UI on video
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, emotion, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

                # Throttle UI-side advice to every 15s
                if time.time() - self.last_advice_time > 15:
                    self.last_advice_time = time.time()
                    # Store for UI thread
                    st.session_state['last_emotion'] = (emotion, self.last_advice_time)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def capture_webcam():
    # STUN server so it works on Streamlit Cloud
    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_config
    )

# -----------------------------
# YouTube embed
# -----------------------------
def stream_video(youtube_link: str):
    st.markdown("#### ▶️ YouTube")
    if "youtube.com" in youtube_link or "youtu.be" in youtube_link:
        vid = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1]
        st.markdown(
            f"""
            <div class="card">
            <iframe width="100%" height="320"
                src="https://www.youtube.com/embed/{vid}"
                title="YouTube video player" frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowfullscreen></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Paste a valid YouTube URL (youtube.com or youtu.be).")

# -----------------------------
# Main App
# -----------------------------
def main():
    if "emotion_counts" not in st.session_state:
        st.session_state.emotion_counts = {e: 0 for e in emotions.values()}
    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = None

    st.markdown("<h1 style='text-align:center'>🎭 AI Face Emotion Recognition (WebRTC)</h1>", unsafe_allow_html=True)
    st.caption("Real-time emotion overlays + advice & quizzes + live trend chart")

    with st.sidebar:
        st.header("🎛️ Controls")
        youtube_link = st.text_input("YouTube link")
        st.markdown("**Legend**")
        for e, c in emotion_colors.items():
            st.markdown(f"<div class='pill' style='background:{c}; margin:3px 6px 0 0'>{e}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])

    # Left: YouTube + Chart
    with col1:
        if youtube_link:
            stream_video(youtube_link)
        else:
            card("No YouTube Video", "Paste a link in the sidebar to play a video alongside the webcam feed.")

        st.markdown("#### 📈 Emotion Trend (session)")
        # Update counts if a new emotion was pushed by the processor
        if st.session_state.last_emotion:
            emotion, ts = st.session_state.last_emotion
            st.session_state.emotion_counts[emotion] += 1
        st.bar_chart(st.session_state.emotion_counts)

    # Right: Webcam + Advice/Quiz
    with col2:
        st.markdown("#### 📷 Live Webcam")
        capture_webcam()

        st.markdown("---")
        if st.session_state.last_emotion:
            emotion, key = st.session_state.last_emotion
            # Colored result card
            bg = emotion_colors.get(emotion, "#ffffff")
            card("Detected Emotion", f"We noticed signs of **{emotion}**.", bg)
            show_emotion_quiz(emotion, str(key))
        else:
            card("Waiting for detection", "Look at the camera to begin.", "#ffffff")

if __name__ == "__main__":
    main()
