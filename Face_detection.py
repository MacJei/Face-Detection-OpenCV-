import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt


#Pre-trained classifier for face detection
path = "Path to the classifier (cv2/data/..)"
face_cascade = cv2.CascadeClassifier(path)

# Read the image
img = cv2.imread(os.getcwd() + 'img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
  )


# ROI for the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Display
cv2.imshow("Faces found" ,img)
cv2.waitKey(0)


