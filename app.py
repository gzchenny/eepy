from flask import Flask, render_template, Response, jsonify
from scripts.camera import generate_frames, data_store, socketio
from flask_socketio import SocketIO
import time
import threading

app = Flask(__name__)
#socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin for testing
socketio.init_app(app, cors_allowed_origins="*")

data_store = {"EAR": 0, "MAR": 0, "is_drowsy": False}


# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get data
@app.route('/data')
def get_data():
    return jsonify(data_store)

def generate_data():
    while True:
        data = {
            "EAR": round(data_store["EAR"], 3),
            "MAR": round(data_store["MAR"], 3),
            "is_drowsy": data_store["is_drowsy"]
        }
        # print("Emitting data:", data)
        socketio.emit("update_data", data)
        time.sleep(1)

thread = threading.Thread(target=generate_data)
thread.daemon = True
thread.start()

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for home page
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)