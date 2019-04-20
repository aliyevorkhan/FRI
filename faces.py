import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read() #capture frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
    	
    	roi_gray = gray[y:y+h, x:x+h] #(ycord_start, ycord_end) -- gray value
    	roi_color = frame[y:y+h, x:x+h] #(ycord_start, ycord_end) -- bgr value   	
    	
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=0 and conf <= 100:
    		
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


    	img_item = "my-image.png" #if find face take that frame 
    	cv2.imwrite(img_item, roi_gray) # ^^and save

    	color = (255, 0, 0) #BGR
    	stroke = 2 # rectangle line thickness
    	end_cord_x = x + w
    	end_cord_y = y +h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) # draw rect with arguments
    	
    #display the resulting frame
    cv2.imshow('frame', frame) #show all frames as video

    #break button declaration
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()
cap.destroyAllWindows()
