import cv2
import numpy as np
from keras.models import model_from_json

def moodModel():
    label = ['angry', 'happy', 'neutral']

    json_file = open("./moodTraining/driver_detect1.json","r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("./moodTraining/driver_detect1.h5")

    image = "./output_image/"  # Replace with the path to your image
    frame = cv2.imread(image)
    frame = cv2.resize(frame, (1180, 660))
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    #print(num_faces)

    for (x, y, w, h) in num_faces:
    #print("face detected!")
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        print("Cropped image shape:", cropped_img.shape)
        # Predict the emotions
        pred = model.predict(cropped_img)
        print("Prediction array:", pred)
        prediction = label[pred.argmax()]
        print("predicted image is ", prediction)
        print(x,y,w,h)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return prediction

