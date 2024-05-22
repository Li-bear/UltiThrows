from flask import Flask, request
from flask import render_template
from flask import Response, jsonify
import cv2
import mediapipe as mp
import forehand_detector as throws_fun
import numpy as np
import time
import random
import math

# Create flask
app = Flask(__name__, static_folder='templates/static')

# static_folder: This parameter allows you to specify the path to the folder where static files (such as CSS, JavaScript, images, etc.) 
# are stored. In the provided code, the static folder is set to 'templates/static', which means that Flask will look for static files in 
# the static folder located within the templates folder.
cap = cv2.VideoCapture(0)

# Set up media pipe
mp_drawing = mp.solutions.drawing_utils
# Import pose estimation model
mp_pose = mp.solutions.pose

# variables for video_data
angle_right = 0
angle_left = 0
error_arm = False
n_throws_global = 0

# exercise reflex
caught_n_frisbee = 0
active_timer = time.time()

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
            image = cv2.flip(image, 1) # mirror effect
            
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
                arm_visible = False
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


def draw_frisbee(n_exercises = 5):
  
    # generate a random x and y position to center the disk
    # define a limit time
    # [x] define a number of throws to practice
    # paint 3, 2, 1 over the screen

    disk_caught = False
    draw_landmarks = False
    global caught_n_frisbee
    caught_n_frisbee = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        active_timer = time.time()

        while cap.isOpened(): #cap.isOpened():
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
            image = cv2.flip(image, 1) # mirror effect

            if not disk_caught:
                width, height = image.shape[1], image.shape[0]
                disk_x = random.randint(math.trunc(width * 0.10), math.trunc(width * 0.90))
                disk_y = random.randint(math.trunc(height * 0.10), math.trunc(height * 0.90))
                disk_caught = True

            
            # Make detection
            results = pose.process(image) # get detection of pose
            
            # Recolor back to BGR for opencv
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                right_wrist_landmarks = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_wrist_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                

                image = cv2.circle(image, (disk_x, disk_y), radius=10, color=(0, 255, 0), thickness=-1)
                
                # check upper hand is right
                r_wrist_x = right_wrist_landmarks[0] * image.shape[1]
                l_wrist_x = left_wrist_landmarks[0] * image.shape[1]
                
                r_wrist_y = right_wrist_landmarks[1] * image.shape[0]
                l_wrist_y = left_wrist_landmarks[1] * image.shape[0]
                
                condition_y_axis_right = r_wrist_y > l_wrist_y
                condition_visibility_points = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.4 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.4
                condition_distance = ((r_wrist_x - l_wrist_x) < 20) and ((r_wrist_x - l_wrist_x) > -20)
                
                # distance to the disk, taking as reference right wrist landmark
                distance_disk_wrist = math.sqrt((r_wrist_x - disk_x)**2 + (r_wrist_y - disk_y)**2)
                condition_disk = distance_disk_wrist < 30
                
                    
                if condition_y_axis_right and condition_visibility_points and condition_distance and condition_disk:
                    disk_x = random.randint(math.trunc(width * 0.10), math.trunc(width * 0.90))
                    disk_y = random.randint(math.trunc(height * 0.10), math.trunc(height * 0.90))
                    caught_n_frisbee += 1
            
            except Exception as e:
                print(f"Error {e}")
                pass

            # Encode the image
            if ret:
                (flag, encodedImage) = cv2.imencode(".jpeg", image)

            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


# TODO: add timer

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


@app.route("/catch_disk_video")
def catch_disk_video():
    return Response(draw_frisbee(), mimetype="multipart/x-mixed-replace; boundary=frame")
        

@app.route("/exercise_reflex")
def exercise_reflex():
    return render_template("reflex_exercise.html")

@app.route('/get_caught_n_frisbee')
def get_caught_n_frisbee():
    return jsonify({"caught_n_frisbee": caught_n_frisbee})

@app.route('/reset_exercise')
def reset_exercise():
    global caught_n_frisbee
    caught_n_frisbee = 0
    return jsonify({"status": "Exercise reset successfully"})

#debug updates programs
if __name__ == "__main__":
    app.run(debug=False)

cap.release()