#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from ultralytics import YOLO
import math
from keras.models import model_from_json

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

label = ['Angry', 'Happy', 'Neutral']
json_file = open("./Module_emotion/driver_detect1.json", "r")
model_json = json_file.read()
json_file.close()
model_emon = model_from_json(model_json)
model_emon.load_weights("./Module_emotion/driver_detect1.h5")

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 28
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("./Module_Drowsiness_Yawn/haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('./Module_Drowsiness_Yawn/shape_predictor_68_face_landmarks.dat')
face_detector = cv2.CascadeClassifier('./Module_emotion/haarcascade_frontalface_default.xml')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=850)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    results = model(frame, stream=True)

    cv2.rectangle(frame, (300,7), (692, 37), (0, 0, 0), 2)
    cv2.putText(frame, "Driver Behavior Monitoring", (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (158, 158, 17), 2)
    cv2.rectangle(frame, (692, 7), (850, 35), (0, 0, 0), 2)
    cv2.rectangle(frame, (692, 35), (850, 65), (0, 0, 0), 2)
    cv2.putText(frame, "EYE: ", (698, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "YAWN: ", (698, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "DROWSINESS ALERT! ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 255,0,0), 2)
    cv2.putText(frame, "YAWN ALERT! ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 255,0,0), 2)
    cv2.putText(frame, "CELL PHONE ALERT! ", (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 255,0,0), 2)
    cv2.putText(frame, "EMOTION! ", (10, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 255,0,0), 2)
    cv2.circle(frame, (284,21), 10, (0,0,255),-1) 
    cv2.circle(frame, (284,53), 10, (0,0,255),-1) 
    cv2.circle(frame, (284,86), 10, (0,0,255),-1) 
    cv2.rectangle(frame, (2,7), (300, 100), (0, 0, 0), 2)
    cv2.rectangle(frame, (2,7), (300, 37), (0, 0, 0), 2)
    cv2.rectangle(frame, (2,70), (300, 100), (0, 0, 0), 2)
    cv2.rectangle(frame, (2,100), (300, 130), (0, 0, 0), 2)
    cv2.rectangle(frame, (170,100), (300, 130), (0, 0, 0), 2)
    cv2.rectangle(frame, (2,7), (265, 100), (0, 0, 0), 2)

    #for rect in rects:
    for (mod1,mod2) in zip(rects,results):
        boxes=mod2.boxes
        x, y, w, h=mod1
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                '''if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)'''
                cv2.circle(frame, (284,21), 10, (0,255,0),-1) 

        else:
            COUNTER = 0
            alarm_status = False
            cv2.circle(frame, (284,21), 10, (0,0,255),-1)

        if (distance > YAWN_THRESH):
                cv2.circle(frame, (284,53), 10, (0,255,0),-1) 
                '''cv2.putText(frame, "Yawn Alert", (10, 160),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()'''
        else:
            alarm_status2 = False
            cv2.circle(frame, (284,53), 10, (0,0,255),-1) 

        cv2.putText(frame, " {:.2f}".format(ear), (760, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, " {:.2f}".format(distance), (760, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)    
        # Extract the region of interest (ROI) and preprocess it
        roi_gray_frame = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # Predict the emotions
        pred = model_emon.predict(cropped_img)
        prediction = label[pred.argmax()]
        # Put the predicted label on the frame
        cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, prediction, (175, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        #boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            #confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])

            # object details
            if (classNames[cls]=="cell phone" or classNames[cls]=="remote"):
                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                #cv2.putText(frame, "CELL PHONE DETECTED", (10, 190),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                cv2.putText(frame,"cell phone", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                cv2.circle(frame, (284,86), 10, (0,255,0),-1) 
            else:
                cv2.circle(frame, (284,86), 10, (0,0,255),-1) 

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
