import cv2
import numpy as np
from keras.models import model_from_json

def moodModel():
    label = ['angry', 'happy', 'neutral']

    # Load the model from JSON file and weights
    json_file = open("driver_detect1.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("driver_detect1.h5")

    # Initialize the face detector
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame and convert to grayscale
        frame = cv2.resize(frame, (1180, 660))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            
            # Extract the region of interest (ROI) and preprocess it
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            pred = model.predict(cropped_img)
            prediction = label[pred.argmax()]

            # Put the predicted label on the frame
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame with the emotion predictions
        cv2.imshow('Mood Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

moodModel()
