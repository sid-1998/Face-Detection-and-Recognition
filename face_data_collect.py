# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np
import os

#Init Camera
cap = cv2.VideoCapture(0)
flag_capturing = False

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './dataset/'
file_name = input("Enter name: ")
while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	
	

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		if flag_capturing == True:
			skip += 1
			if skip%10==0:
				face_data.append(face_section)
				print(len(face_data))


	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1)
	if key_pressed == ord('q'):
		break
	if key_pressed == ord('c'):
		flag_capturing = True
	if skip == 100:
		flag_capturing = False
		break





cap.release()
cv2.destroyAllWindows()


