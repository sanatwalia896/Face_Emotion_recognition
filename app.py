import streamlit as st
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pytube import YouTube
import threading

# Define the Deep Emotion model
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

# Load the trained model
model = Deep_Emotion()
model.load_state_dict(torch.load('Face_Emotion_Recognition_2.pth', map_location=torch.device('cpu')))
model.eval()

# Define the emotions
emotions = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Transform to apply to input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Function to detect faces in an image
def detect_faces(cascade, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to predict emotion from a face image
def predict_emotion(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted.item()]
# Function to show happiness features
def show_happiness_features(unique_key):
    st.write("You seem happy!")
    review = st.text_area("Please write a review of your experience:")
    st.markdown("---")
    st.subheader("Happiness Quiz")
    options = ["Smiling", "Crying", "Frowning", "None"]

    quiz_answer = st.radio("Which of the following is a feature of happiness?", options, key=unique_key, index=None)
    if quiz_answer == "Smiling":
        st.success("Correct! Smiling is a feature of happiness.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show sadness support
def show_sadness_support(unique_key):
    st.write("You seem sad.")
    st.warning("Remember, it's okay to feel sad sometimes. Reach out to someone for support.")
    st.markdown("---")
    st.subheader("Sadness Quiz")
    options = ["Happy", "Gloomy", "Excited", "None"]
    quiz_answer = st.radio("Which of the following is a synonym for 'sad'?", options, key=unique_key, index=None)
    if quiz_answer == "Gloomy":
        st.success("Correct! 'Gloomy' is a synonym for 'sad'.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show a neutral quiz
def show_neutral_quiz(unique_key):
    st.write("You seem neutral.")
    st.markdown("---")
    st.subheader("Neutral Quiz")
    options = ["Indifferent", "None"]
    quiz_answer = st.radio("Which of the following is a synonym for 'neutral'?", options, key=unique_key, index=None)
    if quiz_answer == "Indifferent":
        st.success("Correct! 'Indifferent' is a synonym for 'neutral'.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show anger support
def show_anger_support(unique_key):
    st.write("You seem angry.")
    st.warning("Take a deep breath and try to calm down. Find a way to express your feelings in a healthy manner.")
    st.markdown("---")
    st.subheader("Anger Quiz")
    options = ["Punching a wall", "Taking deep breaths", "Yelling at others", "None"]
    quiz_answer = st.radio("Which of the following is an appropriate way to manage anger?", options, key=unique_key, index=None)
    if quiz_answer == "Taking deep breaths":
        st.success("Correct! Taking deep breaths can help calm down when feeling angry.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show disgust support
def show_disgust_support(unique_key):
    st.write("You seem disgusted.")
    st.warning("Disgust is a normal emotion, but it's important to understand and manage it.")
    st.markdown("---")
    st.subheader("Disgust Quiz")
    options = ["Eating your favorite food", "Seeing something rotten", "Watching a funny movie", "None"]
    quiz_answer = st.radio("Which of the following might elicit a feeling of disgust?", options, key=unique_key, index=None)
    if quiz_answer == "Seeing something rotten":
        st.success("Correct! Seeing something rotten might elicit a feeling of disgust.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show fear support
def show_fear_support(unique_key):
    st.write("You seem fearful.")
    st.warning("It's natural to feel fear, but remember to breathe and focus on what you can control.")
    st.markdown("---")
    st.subheader("Fear Quiz")
    options = ["Avoiding the situation", "Facing the fear gradually", "Ignoring the fear", "None"]
    quiz_answer = st.radio("Which of the following might help overcome fear?", options, key=unique_key, index=None)
    if quiz_answer == "Facing the fear gradually":
        st.success("Correct! Facing the fear gradually can help overcome it.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")

# Function to show surprise support
def show_surprise_support(unique_key):
    st.write("You seem surprised!")
    st.warning("Surprise can be exciting, but take a moment to process and react appropriately.")
    st.markdown("---")
    st.subheader("Surprise Quiz")
    options = ["Gasping", "None"]
    quiz_answer = st.radio("Which of the following is a common reaction to surprise?", options, key=unique_key, index=None)
    if quiz_answer == "Gasping":
        st.success("Correct! Gasping is a common reaction to surprise.")
    elif quiz_answer == "None":
        st.info("No selection made.")
    else:
        st.error("Incorrect. Please try again.")
# Function to capture video from webcam and apply FER model
def capture_webcam():
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to load camera.")

    st.subheader("Webcam Feed")

    # Display the webcam feed
    video_placeholder = st.empty()
    stop_button = st.button("Stop Video Capture")
    advice_time = time.time()
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read frame.")
            break

        faces = detect_faces(face_cascade, frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract face ROI
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size != 0:
                # Predict emotion
                emotion = predict_emotion(face_roi)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if time.time() - advice_time >= 15:
                    advice_time = time.time()
                    if emotion == "Happy":
                        show_happiness_features(advice_time)
                    elif emotion == "Sad":
                        show_sadness_support(advice_time)
                    elif emotion == "Neutral":
                        show_neutral_quiz(advice_time)
                    elif emotion == "Angry":
                        show_anger_support(advice_time)
                    elif emotion == "Disgust":
                        show_disgust_support(advice_time)
                    elif emotion == "Fear":
                        show_fear_support(advice_time)
                    elif emotion == "Surprise":
                        show_surprise_support(advice_time)
# Show popup advising student every 15 seconds
                  
        # Display the frame in Streamlit
        video_placeholder.image(frame, channels="BGR", width=640)

    # Release the VideoCapture object
    cap.release()

# Function to stream YouTube video
def stream_video(youtube_link):
    st.subheader("YouTube Video")
    st.write("Embedding YouTube video...")
    st.write(f"YouTube Video Link: {youtube_link}")
    st.write('<iframe width="640" height="360" src="https://www.youtube.com/embed/' + youtube_link.split('=')[-1] + '" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

def main():
    st.title("Face Emotion Recognition App")
    st.sidebar.subheader("YouTube Video")
    youtube_link = st.sidebar.text_input("Enter YouTube video link:")

    if youtube_link:
        st.subheader("Webcam Feed")
        stream_video(youtube_link)
        capture_webcam()
    else:
        st.warning("Please enter a YouTube video link.")

if __name__ == "__main__":
    main()
