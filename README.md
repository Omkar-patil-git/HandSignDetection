HAND SIGN DETECTION
# 🤟 Sign Language to Text & Speech Converter using OpenCV and Deep Learning

This project is a **real-time Sign Language Recognition system** that translates hand gestures (based on the English alphabet) into **text and speech** using a webcam. It uses **computer vision**, **deep learning**, and **text-to-speech** techniques to assist individuals who are deaf or hard of hearing to communicate more effectively with the hearing world.

## 📸 Demo

<img width="646" height="509" alt="image" src="https://github.com/user-attachments/assets/8a6c107b-727e-4cae-b8e0-0495d2fbb659" />


---

## 🚀 Features

- ✅ Real-time hand gesture recognition using OpenCV and MediaPipe.
- 🔤 Supports 26 alphabet gestures (A-Z).
- ✍️ Forms complete **words** and **sentences**.
- 🗣️ Converts the recognized sentence into **spoken audio** using `pyttsx3`.
- ⌨️ Keyboard-controlled actions:
  - Press **‘a’**: Add the detected letter to current word
  - Press **‘b’**: Add space (end current word)
  - Press **‘s’**: Speak the full sentence aloud
  - Press **‘c’**: Clear all text
  - Press **‘q’**: Quit the program
- 🌈 Colorful and intuitive user interface built with OpenCV drawing tools.
- 📏 Displays the count of words and pending letters.
- 🧠 Uses a trained **CNN model** for alphabet classification.

---

## 🧠 How It Works

1. The webcam captures a video stream and detects a hand using `cvzone`’s `HandTrackingModule`.
2. The hand region is cropped and resized to a fixed size (300x300).
3. A pre-trained Keras model (`keras_model.h5`) predicts the gesture.
4. The corresponding letter is shown on the screen, and the user can confirm it via keypress.
5. The system forms words, builds full sentences, and speaks the result aloud.

---

## 🛠️ Tech Stack

| Tool            | Purpose                                  |
|-----------------|------------------------------------------|
| Python          | Core programming language                |
| OpenCV          | Real-time image processing and UI        |
| MediaPipe       | Hand landmark detection                  |
| cvzone          | Easy wrapper for MediaPipe + OpenCV      |
| TensorFlow / Keras | Model training and gesture recognition |
| pyttsx3         | Text-to-speech conversion                |
| NumPy / Math    | Matrix and image transformations         |

---

## 📂 Project Structure

SignLanguageToSpeech/
 │
 
 ├── Model/
 
 │ ├── keras_model.h5 # Trained hand gesture model
 
 │ └── labels.txt # Corresponding gesture labels
 
 │
 
 ├── test.py # Main application file
 
 ├── README.md # This file
 
 └── requirements.txt # Python dependencies

## 📦 Requirements
opencv-python

 cvzone

 mediapipe

 pyttsx3
 
 numpy

 tensorflow

## 🎯 Use Cases
🧏‍♂️ Helping deaf and mute individuals communicate

🧪 Learning tool for sign language recognition

💬 Gesture-based communication system for silent environments (labs, studios)

🤖 Can be extended into home automation or accessibility apps

## 📈 Future Improvements
Support for Indian Sign Language (ISL)

Support for dynamic gestures (like "Hello", "Thanks")

Add word autocorrection or spell-check

Export conversation to text file

Mobile or web-based version

Enhanced speech options (voice type, pitch, language)

