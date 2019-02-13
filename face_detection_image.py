import cv2
import numpy as np
import os

path = './test_images/'
images = os.listdir(path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

for img in images:
    im = cv2.imread(path+img)
    faces = face_cascade.detectMultiScale(im,1.3,5)

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,0,255), 2)
        # offset = 10
        # face_section = im[y-offset:y+h+offset, x-offset:x+w+offset]

        cv2.imwrite('./output/' + img, im)
        print("Done")
    
