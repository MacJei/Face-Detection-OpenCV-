import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#Classifier
face_cascade = cv2.CascadeClassifier(os.getcwd() + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

while(True):

    # Capture frame-by-frame
    ret, frame = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
      )

    # ROI for the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Release the capture
camera.release()
cv2.destroyAllWindows()