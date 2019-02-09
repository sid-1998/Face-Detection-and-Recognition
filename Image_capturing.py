import cv2
import numpy as np
import os

vc = cv2.VideoCapture(0)
#face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data = []
flag_capturing = False
path = './dataset/1'

skip = 0
while(vc.isOpened()):
    # read image
    rval, frame = vc.read()

    frame = cv2.flip(frame, 1)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    if len(faces) == 0:
        continue
    
    

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        #crop required face: area of interest
        offset = 10# padding of 10 pixels around the face
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        if flag_capturing == True:
            skip +=1
            if skip%10 == 0:

                cv2.imwrite(path + "/" + str(skip)+ ".jpg", face_section)
                face_data.append(face_section)
                print(len(face_data))

            
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)   
    
    keypress = cv2.waitKey(1)
    
    if skip == 100:
        flag_capturing = False
        break
    if keypress == ord('q'):
        break
    elif keypress == ord('c'):
        flag_capturing = True

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

