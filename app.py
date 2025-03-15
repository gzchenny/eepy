from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
from scipy.spatial import distance

app = Flask(__name__)

# Load Face Detector
face_net = cv2.dnn.readNetFromTensorflow("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")

# Load Dlib landmarks predictor
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define landmarks indices
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 50, 52, 54, 56, 58]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[5])
    B = distance.euclidean(mouth[2], mouth[4])
    C = distance.euclidean(mouth[0], mouth[3])
    return (A + B) / (2.0 * C)

# Video streaming generator function
def generate_frames():
    cap = cv2.VideoCapture(0)
    blink_count = 0
    yawn_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Detect face using OpenCV DNN
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_rect = dlib.rectangle(x, y, x1, y1)

                # Detect facial landmarks
                shape = predictor(gray, face_rect)
                landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

                # Calculate EAR for blink detection
                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                # Calculate MAR for yawning detection
                mouth = landmarks[MOUTH]
                mar = mouth_aspect_ratio(mouth)

                # Detect drowsiness
                if ear < 0.33:
                    blink_count += 1
                    if blink_count > 25:
                        cv2.putText(frame, "DROWSY! Wake up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                else:
                    blink_count = 0

                # Detect yawning
                if mar > 0.6:
                    yawn_count += 1
                    if yawn_count > 10:
                        cv2.putText(frame, "Yawning! Take a break!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                else:
                    yawn_count = 0

                # Draw landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Convert frame to bytes for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)