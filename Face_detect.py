import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the cascade classifier (replace with the path to your downloaded file)
face_cascade = cv2.CascadeClassifier("C:\Users\KIIT\Desktop\DA PROJECT\haarcascade_frontalface_alt.xml")

# Define a function to stop the video capture
def stop_stream():
  global running  # Access the global variable
  running = False

# Set a global variable to control the loop
running = True
# Video capture object
cap = cv2.VideoCapture(0)  # 0 for default webcam

while running:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Convert frame to grayscale for face detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  # Draw bounding boxes around detected faces
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  # Display the resulting frame
  cv2.imshow('frame', frame)

  # Check for 'q' key press to exit or 's' to stop the stream
  if cv2.waitKey(1) == ord('q'):
    break
  elif cv2.waitKey(1) & 0xFF == ord('s'):
    stop_stream()  # Call the stop_stream function

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
