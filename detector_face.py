import os
import cv2
import numpy as np
import sys

def dist(x1, x2):
  return np.sqrt(sum(x1 - x2) ** 2)

def knn(train, test, k=5):
  vals = []
  m = train.shape[0]
  for i in range(m):
    ix = train[i, :-1]
    iy = train[i, -1]
    d = dist(test, ix)
    vals.append((d, iy))
  vals_sorted = sorted(vals, key = lambda x: x[0])[:k]
  labels = np.array(vals_sorted)[:, -1]
  output = np.unique(labels, return_counts = True)
  max_fre_index = output[1].argmax()
  pre = output[0][max_fre_index]
  print("pre: ", pre)
  return pre

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
i = 0
face_data = []
data_set = './dataset/'
label = []
class_id = 0
names = {}

for fx in os.listdir(data_set):
  if fx.endswith('.npy'):
    names[class_id] = fx[:-4]
    data_item = np.load(data_set + fx)
    face_data.append(data_item)
    target = class_id * np.ones((data_item.shape[0]))
    class_id += 1
    label.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(label, axis = 0).reshape((-1,1))
trainset = np.concatenate((face_dataset, face_labels), axis = 1)


while True:
  ret, frame = cap.read()
  if ret == False:
    continue
  faces = face_cascade.detectMultiScale(frame, 1.3, 5)
  for face in faces:
    x, y, w, h = face
    offset = 10
    face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
    try:
      face_section = cv2.resize(face_section, (100, 100))
      cv2.imshow("Face Section", face_section)
    except Exception as e:
      print(str(e))
      #sys.exit(1)
    try:
      out = knn(trainset, face_section.flatten())
      pred_name = names[int(out)]
      print(pred_name)
      cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 1)
    except Exception as e:
      print(str(e))
  cv2.imshow("Faces", frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
