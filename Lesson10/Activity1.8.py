import cv2,sys,numpy,os
haar_file="haarcascade_frontalface_default.xml"
datasets="datasets"
path=os.path.join(datasets)
print("Recognizing Face. Please be in sufficient light...")
#Create list of images and list of corresponding names
(images,labels,names,id)=([],[],{},0)
for(subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath + "/" + filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
print(labels)
(images,labels)=[numpy.array(lis) for lis in [images,labels]]
recogniser=cv2.face.LBPHFaceRecognizer_create()
recogniser.train(images,labels)

(width,height)=(130,100)
face_cascade=cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
#capture 30 m of 2


while True:
    return_value, img= webcam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=recogniser.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if prediction[1]>40:
            cv2.putText(img,"% s - %.0f" %(names[prediction[0]], prediction[1]), (x-10,y-10), cv2.FONT_ITALIC,1,(244,255,200),1)
        else:
            cv2.putText(img,"Not Recognised", (x-10,y-10), cv2.FONT_ITALIC,1,(244,255,200),1)
        

    cv2.imshow("OpenCv",img)
    key=cv2.waitKey(10)
    if key==27:
        break
