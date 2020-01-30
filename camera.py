import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
  ret, frame = cap.read()
  faces = face_cascade.detectMultiScale(frame, 1.3, 5)
  # (x, y, w, h)
  if ret == False:
    continue
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
  cv2.imshow("Video Frame", frame)
  key_pressed = cv2.waitKey(1) & 0xFF #11111111
  if key_pressed == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
