from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import face_recognition
import pickle
import base64
from datetime import datetime as dt
from src.models import StudentModel, AttendanceModel
from src.settings import (
    DATASET_PATH,
    HAAR_CASCADE_PATH,
    DLIB_MODEL,
    DLIB_TOLERANCE,
    ENCODINGS_FILE
)

# ====== Flask App Setup ======
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ====== Load Known Encodings from Pickle File ======
with open(ENCODINGS_FILE, "rb") as file:
    data = pickle.loads(file.read())
    known_encodings = data["encodings"]
    known_ids = data["ids"]
    print("[INFO] Face encodings loaded successfully.")

# ====== Helper: Face Recognition & Attendance ======
def recognize_faces_and_mark_attendance(encodings):
    names = []
    known_students = {}

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, DLIB_TOLERANCE)
        display_name = "Unknown"

        if True in matches:
            matched_indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for matched_index in matched_indexes:
                _id = known_ids[matched_index]
                counts[_id] = counts.get(_id, 0) + 1

            _id = max(counts, key=counts.get)
            if _id:
                if _id in known_students:
                    student = known_students[_id]
                else:
                    student = StudentModel.find_by_id(_id)
                    known_students[_id] = student

                if not AttendanceModel.is_marked(dt.today(), student):
                    student_attendance = AttendanceModel(student=student)
                    student_attendance.save_to_db()

                display_name = student.name

        names.append(display_name)

    return names

# ====== Handle Incoming Frame from Client (binary) ======
@socketio.on('client_frame')
def handle_client_frame(data):
    try:
        # Decode bytes -> numpy array
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[ERROR] Frame decoding failed.")
            return

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = frame.shape[1] / float(rgb.shape[1])

        # Face detection and encoding
        boxes = face_recognition.face_locations(rgb, model=DLIB_MODEL)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Face recognition and attendance marking
        names = recognize_faces_and_mark_attendance(encodings)

        # Draw bounding boxes and names
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, str(name), (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Encode frame back to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("[ERROR] Failed to encode frame.")
            return

        # Instead of base64, send raw buffer
        emit('processed_frame', buffer.tobytes(), broadcast=True)

    except Exception as e:
        print("[ERROR]", e)

# ====== Optional CLI Video Attendance Class ======
class VideoAttendanceRecognizer:
    def __init__(self, input_video, app_title="Face Recognition"):
        self.input_video = input_video
        self.app_title = app_title

    def recognize_n_attendance(self):
        print("[INFO] Starting video stream...")
        cap = cv2.VideoCapture(self.input_video)
        known_students = {}

        while True:
            ret, img = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = img.shape[1] / float(rgb.shape[1])
            boxes = face_recognition.face_locations(rgb, model=DLIB_MODEL)
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = recognize_faces_and_mark_attendance(encodings)

            for ((top, right, bottom, left), display_name) in zip(boxes, names):
                if display_name == "Unknown":
                    continue
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(img, display_name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow(f"Recognizing Faces - {self.app_title}", img)
            if cv2.waitKey(100) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Attendance Successful!")
# app = Flask(__name__)
# CORS(app)  # This will allow all domains (development only)
# another route
@app.route('/dashboard',methods=['GET'])
def dashboard():
    students=StudentModel.find_all()
    attendances = AttendanceModel.find_all()
    all_info = []
    for student in students:
        # print(student.name)
        date_time = {
            "dates": []
        }

        for attendance in attendances:
            if student.id == attendance.student_id:
                date_time["dates"].append({
                    "attendance_date": attendance.date.strftime("%Y-%m-%d"),
                    "time":attendance.date.strftime("%I-%M-%p")
                })
        student_data = {
            "id": student.id,
            "name": student.name,
            "date_time": date_time
        }
        
        all_info.append(student_data)
    print(all_info)
    student_json = jsonify(all_info)
    # print(student_json)
    return student_json, 200

# ====== Start Server ======
if __name__ == '__main__':
    print("[INFO] Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000)
