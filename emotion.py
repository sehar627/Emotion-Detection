import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Emotion Model
model = load_model("emotion_model.h5")

emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']

print("Model Loaded Successfully")

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start Webcam
cap = cv2.VideoCapture(0)


def detect_emotion():

    ret, frame = cap.read()

    if not ret:
        return "neutral"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    detected_emotion = "neutral"

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Extract face
        face = gray[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face,(48,48))

        # Normalize
        face = face / 255.0

        # Reshape for CNN
        face = np.reshape(face,(1,48,48,1))

        # Predict emotion
        prediction = model.predict(face)

        emotion_index = np.argmax(prediction)

        detected_emotion = emotions[emotion_index]

        # Display emotion
        cv2.putText(frame, detected_emotion,(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    return detected_emotion


# Run Camera Continuously
while True:

    emotion = detect_emotion()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()