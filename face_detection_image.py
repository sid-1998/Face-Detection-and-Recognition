import cv2
import numpy as np

path = './test_images/dp.png'
img = cv2.imread(path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces = face_cascade.detectMultiScale(img,1.3,5)

for face in faces:
    x,y,w,h = face
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    img_detect = cv2.imwrite('./output/' + "img.jpg", img)