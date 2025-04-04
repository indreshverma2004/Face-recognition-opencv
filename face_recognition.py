import numpy as np
import cv2 as cv
import os
from tkinter import Tk, filedialog
from tkinter import messagebox

# Load the Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# List of people (labels must match your training data)
people = ['Narendra_Modi', 'Barack_Obama', 'Swami_Vivekananda', 'Bill_Gates', 'Elon_Musk']

# Load the trained face recognizer model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Create a Tkinter window (but hide it)
root = Tk()
root.withdraw()
root.update()

# Ask user to select an image file
img_path = filedialog.askopenfilename(
    title='Select an Image for Face Recognition',
    filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')]
)

# Check if user selected a file
if not img_path:
    messagebox.showwarning("No File Selected", "No image selected. Please select an image.")
    exit()

# Load the image
img = cv.imread(img_path)

# Check if image was loaded successfully
if img is None:
    messagebox.showerror("Image Load Error", f"Failed to load image from path: {img_path}")
    exit()

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect face(s)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Process detected faces
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)

    print(f'[INFO] Detected {people[label]} with a confidence of {round(confidence, 2)}')

    # Draw label and rectangle
    cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Show result
cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
