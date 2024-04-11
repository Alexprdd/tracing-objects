import numpy 
import cv2

cars_file="cars.xml"
car_cascade=cv2.CascadeClassifier(cars_file)
webcam=cv2.VideoCapture("video1.avi")

while True:
    return_value, img= webcam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale(gray,1.3,1)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("OpenCv",img)
    key=cv2.waitKey(10)
    if key==27:
        break