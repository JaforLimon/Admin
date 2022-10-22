import cv2

#img = cv2.imread('lina.png')
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
#cv2.imshow("output",img)
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
configPath = './mobilenet.pbtxt'
net = cv2.dnn_DetectionModel(configPath)
net.setInputSize(320,320)
net.setInputScole(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds,confs,bbox = net.detect(img,confThreshold=.5)
    print(classIds,bbox)
    if len(classIds) != 0:
        for classId, confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.recctangle(img,box,color=(0,255,0),thickness=2)
            cv2.putTest(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    cv2.imshow("output",img)
    cv2.waitKey(1)