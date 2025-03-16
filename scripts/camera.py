import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from flask_socketio import SocketIO


socketio = SocketIO()
### to do
# CALIBRATION !!!!!!!!!!!!!!!!

# covering mouth when yawning
# consider glasses detection
# ADJUST THRESHOLDS

# covering mouth when yawning
# consider glasses detection

# 0. if eyes are closed for too long, wake up
# 1. drowsiness eye detection is fixed i think -> check with everyones eyes??
# 2. add detection for coverign hand -> if possible, differentiate between covering mouth and just having it there cus of lauhging or smth
# 3. previous point ties in with 2, emotion detection needs to be added - laughing can close eyes and be mistaken for drowsiness

# OpenCV Face Detector (DNN)
face_net = cv2.dnn.readNetFromTensorflow("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")

# dlib face landmarks
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# indices for eye and mouth landmarks

# 1-indexed
# LEFT_EYE = [37, 38, 39, 40, 41, 42]
# RIGHT_EYE = [43, 44, 45, 46, 47, 48]
# MOUTH = [50, 52, 54, 56, 58, 59]
# 0-indexed
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 50, 52, 54, 56, 58]

# ear_value = 0
# mar_value = 0
data_store = {
    "EAR": 0,
    "MAR": 0,
}

# function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return round(ear, 3)

# function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[5])  # vertical
    B = distance.euclidean(mouth[2], mouth[4])  # vertical
    C = distance.euclidean(mouth[0], mouth[3])  # horizontal
    mar = (A + B) / (2.0 * C)
    return round(mar, 3)

blink_count = 0
yawn_count = 0
frame_counter = 0
update_frequency = 15  # Update UI every 15 frames

def generate_frames():
    global blink_count, yawn_count
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to grayscale
        h, w = frame.shape[:2]

        # detecting face
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_rect = dlib.rectangle(x, y, x1, y1)

                # detecting landmarks facial landmarks
                shape = predictor(gray, face_rect)
                landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

                # EAR calculation for blink detection
                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                ear = round(ear, 3)
                data_store["EAR"] = ear

                # MAR calculation for yawning detection
                mouth = landmarks[MOUTH]
                mar = mouth_aspect_ratio(mouth)
                mar = round(mar, 3)
                data_store["MAR"] = mar

                # sending data to frontend
                data = {"EAR": ear, "MAR": mar}
                socketio.emit("update_data", data)

                # detecting drowsiness
                if ear < 0.3:  # threshold
                    blink_count += 1
                    if blink_count > 15:  # ADJUST THIS
                        cv2.putText(frame, "DROWSY! Wake up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                else:
                    blink_count = 0

                # detecting yawning
                if mar > 0.6:  # threshold
                    yawn_count += 1
                    if yawn_count > 10:  # ADJUST THIS
                        cv2.putText(frame, "Yawning! Take a break!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                else:
                    yawn_count = 0

                # drawing landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # if frame_counter >= update_frequency:
        #     data = {"EAR": ear, "MAR": mar}
        #     socketio.emit("update_data", data)
        #     frame_counter = 0
        
        # converting frame to bytes for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

cv2.destroyAllWindows()