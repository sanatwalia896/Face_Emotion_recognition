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

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #2c3e50;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .emotion-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .video-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .iframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

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
        x = self.dropdown(x)
        x = self.fc3(x)
        return x

model = Deep_Emotion()
model.load_state_dict(torch.load('Face_Emotion_Recognition_2.pth', map_location=torch.device('cpu')))
model.eval()

emotions = {
    0: "Angry 😣", 1: "Disgust 😖", 2: "Fear 😨", 3: "Happy 😊",
    4: "Sad 😔", 5: "Surprise 😲", 6: "Neutral 😐"
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Placeholder for UI functions (assuming they exist)
def show_happiness_features(key):
    st.markdown(f"<div class='emotion-card'>😊 Happiness detected! Enjoy some positive content!</div>", unsafe_allow_html=True)

def show_sadness_support(key):
    st.markdown(f"<div class='emotion-card'>😔 Feeling sad? Here's some support for you.</div>", unsafe_allow_html=True)

def show_neutral_quiz(key):
    st.markdown(f"<div class='emotion-card'>😐 Neutral vibe? Try this fun quiz!</div>", unsafe_allow_html=True)

def show_anger_support(key):
    st.markdown(f"<div class='emotion-card'>😠 Feeling angry? Let's cool down with some tips.</div>", unsafe_allow_html=True)

def show_disgust_support(key):
    st.markdown(f"<div class='emotion-card'>😖 Disgusted? Here's something to lighten the mood.</div>", unsafe_allow_html=True)

def show_fear_support(key):
    st.markdown(f"<div class='emotion-card'>😨 Feeling scared? Find some calming resources here.</div>", unsafe_allow_html=True)

def show_surprise_support(key):
    st.markdown(f"<div class='emotion-card'>😲 Surprised? Check out this exciting content!</div>", unsafe_allow_html=True)

# Emotion UI display
def show_emotion_quiz(emotion, key):
    emotion_functions = {
        "Happy 😊": show_happiness_features,
        "Sad 😔": show_sadness_support,
        "Neutral 😐": show_neutral_quiz,
        "Angry 😣": show_anger_support,
        "Disgust 😖": show_disgust_support,
        "Fear 😨": show_fear_support,
        "Surprise 😲": show_surprise_support
    }
    if emotion in emotion_functions:
        emotion_functions[emotion](key)

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

# Emotion predictor
def predict_emotion(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted.item()]

# Webcam capture
def capture_webcam():
    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

# YouTube video embed
def stream_video(youtube_link):
    video_id = youtube_link.split("v=")[-1].split("&")[0] if "v=" in youtube_link else youtube_link.split("/")[-1]
    st.markdown(f"""
        <div class='iframe-container'>
            <iframe width="100%" height="360" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>
        </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    st.markdown("<h1 class='title'>Face Emotion Recognition App</h1>", unsafe_allow_html=True)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Control Panel")
        youtube_link = st.text_input("Enter YouTube Video Link:", placeholder="e.g., https://www.youtube.com/watch?v=...")
        if youtube_link:
            if st.button("Load Video"):
                st.session_state['youtube_link'] = youtube_link

    # Main content
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        
        # Layout with columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Webcam Feed")
            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
            capture_webcam()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Detected Emotion")
            if 'last_emotion' in st.session_state:
                emotion, key = st.session_state['last_emotion']
                show_emotion_quiz(emotion, str(key))
            else:
                st.markdown("<div class='emotion-card'>Waiting for emotion detection...</div>", unsafe_allow_html=True)

        # Display YouTube video if link is provided
        if 'youtube_link' in st.session_state:
            st.subheader("Recommended Video")
            stream_video(st.session_state['youtube_link'])
        else:
            st.warning("Enter a YouTube link in the sidebar to watch a video.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()