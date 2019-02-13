import cv2
from classification_model import knn



mapping = {
    0: "Siddharth",
    1: "Thor"
}
vc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    
    rval,frame = vc.read()
    frame = cv2.flip(frame, 1)
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    if(len(faces)==0):
        continue

    for face in faces:
        
        x,y,w,h = face

        #Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_section = face_section.reshape(1,-1)
        print(face_section.shape)
        #Predicted Label (out)
        out = knn.predict(face_section)
        print(out)
        #Display on the screen the name and rectangle around it
        pred_name = mapping[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key = cv2.waitKey(1)
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



