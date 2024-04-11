import numpy
import cv2

human_file=cv2.imread("Image2.jpg")
human_detection=cv2.HOGDescriptor()
human_detection.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

humans,return_value=human_detection.detectMultiScale(human_file,winStride=(10,10),padding=(32,32),scale=1.1)
print(humans)
for (x,y,w,h) in humans:
    cv2.rectangle(human_file,(x,y),(x+w,y+h),(255,0,0),2)

cv2.putText(human_file,"Humans ="+ str(len(humans)),(10,40),cv2.FONT_ITALIC,2,(255,255,200),2)
cv2.imshow("humans", human_file)
cv2.waitKey(0)
