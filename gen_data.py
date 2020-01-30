import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
i = 0
face_data = []
data_set = './dataset/'
file_name = input("Enter person's name: ")


while True:

  ret, frame = cap.read()
  faces = face_cascade.detectMultiScale(frame, 1.3, 5)
  faces = sorted(faces, key = lambda f: f[2] * f[3], reverse = True)
  # (x, y, w, h)
  if ret == False:
    continue
  for (x, y, w, h) in faces[-1:]:
    offset = 10
    face_section = frame[y - offset: y + h + offset, x - offset: x + w + offset]
    try:
      face_section = cv2.resize(face_section, (100, 100))
      cv2.imshow("Face Section", face_section)
    except Exception as e:
      # print(str(e))
      print("************error**************")
      sys.exit()
    i = i + 1
    if i % 10 == 0:
      face_data.append(face_section)
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 2)
  cv2.imshow("Video Frame", frame)
  key_pressed = cv2.waitKey(1) & 0xFF #11111111
  if len(face_data) == 3:
    break
  if key_pressed == ord('q'):
    break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data)
np.save(data_set + file_name + '.npy', face_data)
print(face_data.shape)

cap.release()
cv2.destroyAllWindows()
