from flask import Flask, render_template, Response, jsonify
from scripts.camera import generate_frames, data_store, socketio
from flask_socketio import SocketIO
import time
import threading
from scripts.ai.ai_agent import main  # Import the main function from ai_agent

app = Flask(__name__)
#socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin for testing
socketio.init_app(app, cors_allowed_origins="*")

data_store = {"EAR": 0, "MAR": 0, "is_drowsy": False}


# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# New route for AI output
@app.route('/ai_output')
def ai_output():
    result = main()  # Call the main function from ai_agent
    socketio.emit("chat_message", {"message": result})  # Emit the result to the chatbox
    return jsonify({"message": result})  # Return the result as JSON

# Route to get data
@app.route('/data')
def get_data():
    return jsonify(data_store)

def generate_data():
    while True:
        data = {
            "EAR": f"{data_store['EAR']:.3f}",
            "MAR": f"{data_store['MAR']:.3f}",
            # 'EAR': 0.123,
            # 'MAR': 0.456,
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