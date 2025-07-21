import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Optional: Adjust speaking speed



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Word formation variables
detected_letters = []
current_word = ""
full_sentence = ""
pending_letter = None

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # Create blank white image at start of each frame
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) // 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) // 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            pending_letter = labels[index]

            # Draw pending letter (yellow border means unconfirmed)
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (0, 255, 255), cv2.FILLED)
            cv2.putText(imgOutput, pending_letter,
                        (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)

            # Bounding box (yellow for unconfirmed)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 255), 4)

        except Exception as e:
            print(f"Error processing hand: {e}")

    # Current word display
    cv2.rectangle(imgOutput, (0, 0), (img.shape[1], 70), (0, 0, 0), cv2.FILLED)
    cv2.putText(imgOutput, f"Word: {current_word}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(imgOutput, f"Sentence: {full_sentence.strip()}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Controls instructions (more subtle)
    cv2.putText(imgOutput, "'a'=Add  'b'=Space  's'=Speak  'c'=Clear  'q'=Quit", (10, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show image windows
    if imgCrop is not None and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
        cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    if key == ord('q'):  # Quit
        break

    elif key == ord('c'):  # Clear everything
        detected_letters = []
        current_word = ""
        full_sentence = ""

    elif key == ord('a') and pending_letter is not None:  # Add letter to current word
        detected_letters.append(pending_letter)
        current_word = ''.join(detected_letters)

    elif key == ord('b'):  # Add space, end word
        if current_word:
            full_sentence += current_word + " "
            detected_letters = []
            current_word = ""

    elif key == ord('s'):  # Speak sentence
        # Always update sentence with current word if any
        if current_word:
            full_sentence += current_word + " "
            current_word = ""
            detected_letters = []

        sentence_to_speak = full_sentence.strip()

        if sentence_to_speak:
            print(f"Speaking: {sentence_to_speak}")
            engine.say(sentence_to_speak)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()