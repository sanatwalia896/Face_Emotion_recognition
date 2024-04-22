# Face Emotion Recognition Model

## Overview
This repository contains code for a Face Emotion Recognition model using the Fer 2013 dataset. Additionally, it includes a Streamlit app that enables users to stream YouTube videos and analyze the emotions of individuals viewing those videos through their webcam. The app is designed with a focus on students and teachers, allowing teachers to upload their videos to YouTube and monitor how students react to them.

## Features
- **Face Emotion Recognition Model**: Trained on the Fer 2013 dataset, the model can detect emotions including happiness, sadness, anger, surprise, fear, disgust, and neutral expressions.
- **Streamlit App**: Users can stream YouTube videos and analyze the emotions of viewers in real-time through their webcam.
- **Teacher-Student Interaction**: Teachers can upload their videos to YouTube and observe students' reactions to gauge engagement and emotional response.

## Dataset
The model is trained on the Fer 2013 dataset, which contains grayscale images of faces labeled with seven different emotions. 

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Streamlit
- YouTube Data API

## Installation
1. Clone the repository:
   ```
   [git clone https://github.com/yourusername/facial-emotion-recognition.git](https://github.com/sanatwalia896/My_Projects)
   cd new4.py
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up YouTube Data API credentials. Instructions can be found [here](https://developers.google.com/youtube/registering_an_application).

## Usage
1. Train the model using `train_model.py`.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. In the app, enter the YouTube video URL and click on the 'Start Emotion Analysis' button.
4. Allow webcam access when prompted.
5. The app will display the video with real-time emotion analysis overlaid on each face detected.

## Contributors
- [Sanat Walia ](https://github.com/sanatwalia896)


## Acknowledgements
- Fer 2013 dataset: [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Streamlit: [Streamlit](https://streamlit.io/)
- TensorFlow: [TensorFlow](https://www.tensorflow.org/)
- OpenCV: [OpenCV](https://opencv.org/)

## Get Involved
Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/yourusername/facial-emotion-recognition/issues) if you want to contribute.
