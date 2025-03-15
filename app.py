from flask import Flask, render_template, Response, jsonify
from scripts.camera import generate_frames, data_store
from flask_socketio import SocketIO
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin for testing

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
        data = {"EAR": data_store["EAR"], "MAR": data_store["MAR"]}  # GET MORE DATA
        print("Emitting data:", data)
        socketio.emit("update_data", data)
        time.sleep(1)

thread = threading.Thread(target=generate_data)
thread.daemon = True
thread.start()

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)