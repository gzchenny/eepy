from flask import Flask, render_template, Response
from scripts.camera import generate_frames
from flask_socketio import SocketIO
import random
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin for testing

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_data():
    while True:
        data = {"value": random.randint(1, 100)}
        print("Emitting data:", data)  # Print data to console for verification
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