from flask import Flask
from flask import render_template
from flask import Response, jsonify
import cv2
import mediapipe as mp
import forehand_detector as throws_fun
import numpy as np

# Create flask
#app = Flask(__name__)
app = Flask(__name__, static_folder='templates/static')
cap = cv2.VideoCapture(0)


# Set up media pipe
mp_drawing = mp.solutions.drawing_utils
# Import pose estimation model
mp_pose = mp.solutions.pose

# variables for video_data
angle_right = None
angle_left = None
error_arm = False

def generate():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Read a frame from the video
            # ret -> return variable
            # frame -> image from cap
            ret, frame = cap.read()

            if not ret:
                print("End of video.")
                break  # Exit the loop when the video is finished.
            
            # Detect stuff and render
            
            # Recolor image
            # default of opencv is rgb, that's why we recolor
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # set the image to not be modified and save memory
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image) # get detection of pose
            
            # Recolor back to BGR for opencv
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            global angle_right, angle_left, error_arm

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # calculate angle
                angle_right = throws_fun.calculate_angle(shoulder_right, elbow_right, wrist_right)
                angle_left = throws_fun.calculate_angle(shoulder_left, elbow_left, wrist_left)
                
            except Exception as e:
                print(f"Error: {e}")
                pass
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if angle_right < 90:
                error_arm = True
                #cv2.putText(image, "Estira el codo", tuple(np.multiply(elbow_right, [100, 200]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if ret:
                (flag, encodedImage) = cv2.imencode(".jpeg", image)

            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route("/video_data")
def video_data():
    return {"angle_right" : angle_right, 'error_arm': error_arm, 'angle_left': angle_left}

# principal route
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")

@app.route("/exercise_forehand")
def exercise_forehand():
    return render_template("forehand_exercise.html")

#video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

#debug updates programs
if __name__ == "__main__":
    app.run(debug=True)

cap.release()