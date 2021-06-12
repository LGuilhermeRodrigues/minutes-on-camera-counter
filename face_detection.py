import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

from datetime import datetime

before = datetime.now()
lastSeen = datetime.now()
count = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.9) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      print("Trying to open camera again.")
      cap = cv2.VideoCapture(0)
      continue

    now = datetime.now()
    if now.minute != before.minute:
      if before.minute == lastSeen.minute and before.hour == lastSeen.hour:
        count+=1
    before = now


    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
        lastSeen = datetime.now()
    cv2.putText(image,f'Minutes: {count}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

    

cap.release()