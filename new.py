import cv2 as cv
import numpy as np
from time import sleep
import smtplib
import imghdr
from email.message import EmailMessage

Sender_Email = "vijaysakre2000@gmail.com"
Reciever_Email = "ullasbc11@gmail.com"
Password = "vinayaka123"

key = cv.waitKey(1)
webcam = cv.VideoCapture(0)
sleep(2)

cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.4

#### LOAD MODEL
## Coco Names
classesFile = "classes.names"
classNames = []
with open(classesFile, 'rt') as f: #open the file and read in text mode
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

## Model Files
modelConfiguration = "yolov3tiny.cfg"
modelWeights = "yolov3-tiny.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#Take an image,save and send
def Sendmail(frame):
    cv.imwrite(filename='saved_img.jpg', img=frame)
    newMessage = EmailMessage()
    newMessage['Subject'] = "Check out the animal"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('Action should be taken ASAP!')

    with open('saved_img.jpg', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(newMessage)

#Function for performing specific task
def Task(fname,frame):
    # print(fname)
    fname=fname.upper()
    if(fname=="TIGER" or fname=="LION" or fname=="LEOPARD"):
        print("Take a picture and inform to forest officer via mail")
        Sendmail(frame)
    elif(fname=="ELEPHANT"):
        print("bee sound buzzzz buzzz")
    elif(fname=="MONKEY"):
        print("ROARING grraaaauuuu!")
    else:
        print("Caught bear")


#if the probability of given object is good calling function
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = [] #bounding box contains x,y,width,height
    classIds = [] #contains all class IDs
    confs = [] #confidence values
    for output in outputs:
        for det in output:
            scores = det[5:] #remove first 5 elements
            classId = np.argmax(scores) #max score in the object
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT) #width and height of an image
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold) #eliminates overlapping boxes

    for i in indices:
        i = indices[0]
        box = bbox[indices[0]]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) #bounding box
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #print(classNames[classIds[i]])
        Task(classNames[classIds[i]],img)


while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #image is converted to blob format
    net.setInput(blob)
    layersNames = net.getLayerNames()
    # print(layersNames)
    #85 specifications will be having in layer
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv.imshow('Image', img)
    cv.waitKey(1)