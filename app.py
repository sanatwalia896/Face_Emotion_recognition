import streamlit as st
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------- Custom CSS ----------------------
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
    .iframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Model Definition ----------------------
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

# Load model
model = Deep_Emotion()
model.load_state_dict(torch.load('Face_Emotion_Recognition_2.pth', map_location=torch.device('cpu')))
model.eval()

# ---------------------- Helpers ----------------------
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

def predict_emotion(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted.item()]

# ---------------------- Emotion Cards ----------------------
def show_card(message, emoji):
    st.markdown(f"<div class='emotion-card'>{emoji} {message}</div>", unsafe_allow_html=True)

def show_emotion_quiz(emotion):
    cards = {
        "Happy 😊": ("Happiness detected! Enjoy some positive content!", "😊"),
        "Sad 😔": ("Feeling sad? Here's some support for you.", "😔"),
        "Neutral 😐": ("Neutral vibe? Try this fun quiz!", "😐"),
        "Angry 😣": ("Feeling angry? Let's cool down with some tips.", "😣"),
        "Disgust 😖": ("Disgusted? Here's something to lighten the mood.", "😖"),
        "Fear 😨": ("Feeling scared? Find some calming resources here.", "😨"),
        "Surprise 😲": ("Surprised? Check out this exciting content!", "😲")
    }
    if emotion in cards:
        msg, emoji = cards[emotion]
        show_card(msg, emoji)

# ---------------------- Camera Input (Streamlit Cloud Safe) ----------------------
def capture_with_camera():
    img_file = st.camera_input("Take a picture with your webcam")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.warning("No face detected. Try again!")
        else:
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                if roi.size != 0:
                    emotion = predict_emotion(roi)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    st.session_state['last_emotion'] = emotion

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Detected Emotion(s)", use_column_width=True)

# ---------------------- YouTube Embed ----------------------
def stream_video(youtube_link):
    video_id = youtube_link.split("v=")[-1].split("&")[0] if "v=" in youtube_link else youtube_link.split("/")[-1]
    st.markdown(f"""
        <div class='iframe-container'>
            <iframe width="100%" height="360" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>
        </div>
    """, unsafe_allow_html=True)

# ---------------------- Main App ----------------------
def main():
    st.markdown("<h1 class='title'>Face Emotion Recognition App</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        youtube_link = st.text_input("Enter YouTube Video Link:", placeholder="e.g., https://www.youtube.com/watch?v=...")
        if youtube_link:
            st.session_state['youtube_link'] = youtube_link

    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📸 Capture with Camera")
            capture_with_camera()

        with col2:
            st.subheader("Detected Emotion")
            if 'last_emotion' in st.session_state:
                show_emotion_quiz(st.session_state['last_emotion'])
            else:
                st.markdown("<div class='emotion-card'>Waiting for emotion detection...</div>", unsafe_allow_html=True)

        if 'youtube_link' in st.session_state:
            st.subheader("Recommended Video")
            stream_video(st.session_state['youtube_link'])
        else:
            st.warning("Enter a YouTube link in the sidebar to watch a video.")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
